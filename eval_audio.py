from scipy.signal import find_peaks
from scipy.stats import linregress
from libs.SpeechGenderCls import get_gender

from eval import *


class ASound(EvalUnit):
    inst_name = 'a_sound'

    @abstractmethod
    def _evaluate(self):
        pass

    def evaluate(self):
        audio_list, label_list, human_eval_res_list, idx_list = self._evaluate()

        # # CLAPScore audio-text
        # scores = compute_clapscore_at(audio_list, label_list)
        # for idx, data in enumerate(self.res_list):
        #     data['model_eval'] = [scores[i] if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        # self.save()

        # # Gemini-2.0
        # query_list = [form_gemini_mm_query(f"Does the given audio obviously contain the sound of {l}? Explain step "
        #                                    f"by step and end your answer with \"Yes\" or \"No\".", audios=[a])
        #               for a, l in zip(audio_list, label_list)]
        # responses = batch(query_gemini, query_list, model='gemini-2.5-pro-preview-03-25', temperature=0.0, num_worker=1)
        # for idx, data in enumerate(self.res_list):
        #     data['model_eval'] = [float('yes' in responses[i].strip().lower()[-20:]) if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        # self.save()

        # CLAPScore audio-audio
        ref_audio_map = pd.read_csv('./datasets/ESC-50/dataset.csv')
        scores = []
        for audio, label in zip(audio_list, label_list):
            file_list = ref_audio_map[ref_audio_map['category'] == label]['filename'].tolist()
            ref_audio_list = []
            for file_dir in file_list:
                ref_audio, sr = librosa.load('./datasets/ESC-50/audio/' + file_dir)
                if sr != SAMPLE_RATE:
                    ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                ref_audio_list.append(ref_audio)
            scores.append(compute_clapscore_aa(audio, ref_audio_list))
        for idx, data in enumerate(self.res_list):
            data['model_eval'] = [scores[i] if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        self.save()

    def human_evaluate(self):
        audio_list, label_list, human_eval_res_list, idx_list = self._evaluate()
        if len(audio_list) == 0:
            for data in self.res_list:
                data['human_eval'] = [0.0]
            return
        interface = MultiLabelInterface(
            label_list=['Yes', 'No'],
            eval_inst_list=[f"Is the given audio about {label}?" for label in label_list],
            data_list=human_eval_res_list,
            mm_type='a'
        )
        interface.start()
        for idx, data in enumerate(self.res_list):
            data['human_eval'] = [1.0 - float(interface.eval_list[i]) if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        self.save()

    def compute_accuracy(self, threshold=0.68, return_list=False):
        model_eval_list = [res['model_eval'] for res in self.res_list]
        model_eval_list = [np.mean([e > threshold for e in model_eval]) for model_eval in model_eval_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self, threshold=0.68):
        model_eval_list = [res['model_eval'] for res in self.res_list]
        human_eval_list = [res['human_eval'] for res in self.res_list]

        # # Optimal threshold
        # model_eval_cat = list(itertools.chain(*model_eval_list))
        # human_eval_cat = list(itertools.chain(*human_eval_list))
        # threshold = find_optimal_threshold(model_eval_cat, human_eval_cat)

        model_eval_list = [np.mean([e > threshold for e in model_eval]) for model_eval in model_eval_list]
        human_eval_list = [np.mean(human_eval) for human_eval in human_eval_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class ASoundBeginEnd(ASound):
    inst_name = 'a_sound_begin_end'

    def _evaluate(self):
        audio_list = []
        label_list = []
        human_eval_res_list = []
        idx_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            idx_list.append([])
            if 'start' in inst:
                if len(data['audio_list']) != 1:
                    idx_list.append(FAILED_TOKEN)
                else:
                    sample_size = min(4, len(data['audio_list'][0]) // (SAMPLE_RATE * 2))
                    audio_list.append(data['audio_list'][0][: SAMPLE_RATE * sample_size])
                    label_list.append(inst['start'])
                    human_eval_res_list.append({'query': data['query'], 'audio_list': [audio_list[-1]]})
                    idx_list[-1].append(idx)
                    idx += 1
            if 'end' in inst:
                if len(data['audio_list']) != 1:
                    idx_list.append(FAILED_TOKEN)
                else:
                    sample_size = min(4, len(data['audio_list'][0]) // (SAMPLE_RATE * 2))
                    audio_list.append(data['audio_list'][0][- SAMPLE_RATE * sample_size:])
                    label_list.append(inst['end'])
                    human_eval_res_list.append({'query': data['query'], 'audio_list': [audio_list[-1]]})
                    idx_list[-1].append(idx)
                    idx += 1
        return audio_list, label_list, human_eval_res_list, idx_list


class ASoundInclude(ASound):
    inst_name = 'a_sound_include'

    def _evaluate(self):
        audio_list = []
        label_list = []
        human_eval_res_list = []
        idx_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                idx_list.append([FAILED_TOKEN])
                continue
            begin = round(inst['range'][0] * len(data['audio_list'][0]))
            end = round(inst['range'][1] * len(data['audio_list'][0]))
            audio_list.append(data['audio_list'][0][begin: end])
            label_list.append(inst['target'])
            human_eval_res_list.append({'query': data['query'], 'audio_list': [audio_list[-1]]})
            idx_list.append([idx])
            idx += 1
        return audio_list, label_list, human_eval_res_list, idx_list


class ASoundCoT(ASound):
    inst_name = 'a_sound_cot'

    def _evaluate(self):
        audio_list = []
        label_list = []
        human_eval_res_list = []
        idx_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                idx_list.append([FAILED_TOKEN])
                continue
            audio_list.append(data['audio_list'][0])
            label_list.append(inst['target'])
            human_eval_res_list.append(data)
            idx_list.append([idx])
            idx += 1
        return audio_list, label_list, human_eval_res_list, idx_list


class ASoundSilence(ASound):
    inst_name = 'a_sound_silence'

    def _evaluate(self):
        audio_list = []
        label_list = []
        human_eval_res_list = []
        idx_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                idx_list.append([FAILED_TOKEN, FAILED_TOKEN])
                continue
            audio_segs = audio_segmentation(data['audio_list'][0])
            if len(audio_segs) != 2:
                idx_list.append([FAILED_TOKEN, FAILED_TOKEN])
                continue
            audio_list += audio_segs
            label_list += [inst['start'], inst['end']]
            human_eval_res_list.append({'query': data['query'], 'audio_list': [audio_segs[0]]})
            human_eval_res_list.append({'query': data['query'], 'audio_list': [audio_segs[1]]})
            idx_list.append([idx, idx + 1])
            idx += 2
        return audio_list, label_list, human_eval_res_list, idx_list


class ASpeechAttribute(EvalUnit):
    inst_name = 'a_speech_attribute'
    language = 'english'

    def evaluate(self):
        res_list = []
        inst_list = []
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['auto_eval'] = [FAILED_TOKEN, FAILED_TOKEN, FAILED_TOKEN, FAILED_TOKEN]
                data['auto_eval_score'] = [0.0, 0.0, 0.0, 0.0]
                data['transcript'] = ''
                data['wer'] = 0.0
                continue
            res_list.append(data)
            inst_list.append(inst)
        transcripts, wers = transcribe_speech(
            [data['audio_list'][0] for data in res_list],
            [inst['text'] for inst in inst_list],
            self.language
        )
        for trans, wer, data in zip(transcripts, wers, res_list):
            data['transcript'] = trans
            data['wer'] = wer
        self.save()

        genders = get_gender([f'./output/{self.model_name}/audio/{self.inst_name}_{i}.wav' for i in range(len(res_list))])
        for data, inst, gender in zip(res_list, inst_list, genders):
            data['auto_eval'] = [gender, FAILED_TOKEN, FAILED_TOKEN, FAILED_TOKEN]
            data['auto_eval_score'] = [float(gender == ('male', 'female').index(inst['gender'])), FAILED_TOKEN, FAILED_TOKEN, FAILED_TOKEN]
        self.save()

        for data, inst in zip(res_list, inst_list):
            pitch, pitch_s = calculate_pitch(data['audio_list'][0], data['auto_eval'][0], inst)
            if data['auto_eval'][0] == 0:
                data['auto_eval'][1] = pitch
                data['auto_eval_score'][1] = pitch_s
            else:
                data['auto_eval'][2] = pitch
                data['auto_eval_score'][2] = pitch_s
            speed, speed_s = calculate_speed(data['audio_list'][0], data['transcript'], inst, self.language)
            data['auto_eval'][3] = speed
            data['auto_eval_score'][3] = speed_s
        self.save()

    def human_evaluate(self):
        human_data_list = []
        idx = 0
        for data in self.res_list:
            if len(data['audio_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                data['human_eval_score'] = 0.0
                continue
            human_data_list.append(data)
            data['human_eval'] = idx
            idx += 1

        interface = MultiLabelInterface(
            label_list=['Male', 'Female', 'None of the above'],
            eval_inst_list=["What is the gender of the speaker in the given speech?"] * len(human_data_list),
            data_list=human_data_list,
            mm_type='a'
        )
        interface.start()
        for data, inst in zip(self.res_list, self.inst_list):
            data['human_eval'] = interface.eval_list[data['human_eval']]
            data['human_eval_score'] = float(data['human_eval'] == ('male', 'female').index(inst['gender'])) \
                if isinstance(data['human_eval'], int) else data['human_eval']
        self.save()

    def compute_accuracy(self, return_list=False):
        wer_list = [data['wer'] for data in self.res_list]
        model_eval_list = [np.prod([me for me in data['auto_eval_score'] if me != FAILED_TOKEN]) for data in self.res_list]
        combined_list = [a * m for a, m in zip(wer_list, model_eval_list)]
        if return_list:
            return combined_list
        return np.mean(combined_list)

    def compute_correlation(self):
        model_eval_list = [res['auto_eval_score'][0] for res in self.res_list if res['human_eval'] <= 1]
        human_eval_list = [res['human_eval_score'] for res in self.res_list if res['human_eval'] <= 1]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class ASpeechChinese(ASpeechAttribute):
    inst_name = 'a_speech_chinese'
    language = 'chinese'


class ASpeechImitate(EvalUnit):
    inst_name = 'a_speech_imitate'

    def evaluate(self):
        audio_list = []
        text_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['transcript'] = FAILED_TOKEN
                continue
            audio_list.append(data['audio_list'][0])
            text_list.append(inst['text'])
            data['transcript'] = idx
            idx += 1
        transcripts, wers = transcribe_speech(audio_list, text_list)
        for data in self.res_list:
            data['wer'] = wers[data['transcript']] if data['transcript'] != FAILED_TOKEN else 0.0
            data['transcript'] = transcripts[data['transcript']] if data['transcript'] != FAILED_TOKEN else ''

        self.load_inst_mm()
        audio_list = []
        ref_audio_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['model_eval'] = FAILED_TOKEN
                continue
            audio_list.append(data['audio_list'][0])
            ref_audio_list.append(inst['audio_list'][0])
            data['model_eval'] = idx
            idx += 1
        sim_scores = calculate_speech_similarity(audio_list, ref_audio_list)
        for data in self.res_list:
            data['model_eval'] = sim_scores[data['model_eval']] if data['model_eval'] != FAILED_TOKEN else 0.0
        self.save()

    def human_evaluate(self):
        self.load_inst_mm()
        ref_audio_list = []
        human_data_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            ref_audio_list.append(inst['audio_list'][0])
            human_data_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=['Are the speeches coming from the same speaker?'] * len(ref_audio_list),
            data_list=human_data_list,
            ref_list=ref_audio_list,
            mm_type='a'
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = float(interface.eval_list[data['human_eval']] == 0) if data['human_eval'] != FAILED_TOKEN else 0.0
        self.save()

    def compute_accuracy(self, threshold=0.93, return_list=False):
        wer_list = [data['wer'] for data in self.res_list]
        model_eval_list = [data['model_eval'] for data in self.res_list]
        model_eval_list = [float(model_eval > threshold) for model_eval in model_eval_list]
        combined_list = [a * m for a, m in zip(wer_list, model_eval_list)]
        if return_list:
            return combined_list
        return np.mean(combined_list)

    def compute_correlation(self, threshold=0.93):
        model_eval_list = [data['model_eval'] for data in self.res_list]
        human_eval_list = [data['human_eval'] for data in self.res_list]

        # # Optimal threshold
        # threshold = find_optimal_threshold(model_eval_list, human_eval_list)

        model_eval_list = [float(model_eval > threshold) for model_eval in model_eval_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


class ASpeechModify(EvalUnit):
    inst_name = 'a_speech_modify'

    def evaluate(self):
        audio_list = []
        idx = 0
        for data in self.res_list:
            if len(data['audio_list']) != 1:
                data['transcript'] = FAILED_TOKEN
                continue
            audio_list.append(data['audio_list'][0])
            data['transcript'] = idx
            idx += 1
        transcripts, _ = transcribe_speech(audio_list)
        for data in self.res_list:
            data['transcript'] = transcripts[data['transcript']] if data['transcript'] != FAILED_TOKEN else ''
        self.save()

        scores = text_instruction_following_verify(
            [data['transcript'] for data in self.res_list],
            [inst['constraint'] for inst in self.inst_list]
        )
        for score, data in zip(scores, self.res_list):
            data['auto_eval'] = score
        self.save()

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


class ASpeechConstraint(ASpeechModify):
    inst_name = 'a_speech_constraint'


class AMusicAttribute(EvalUnit):
    inst_name = 'a_music_attribute'

    def evaluate_instrument(self):
        res_list = []
        inst_list = []
        for data, inst in zip(self.res_list, self.inst_list):
            if 'instrument' in inst:
                res_list.append(data)
                inst_list.append(inst)
        if len(inst_list) == 0:
            return

        # # ClapScore audio-text
        # labels = [inst['instrument'] + ' music' for inst in inst_list]
        # scores = compute_clapscore_at([data['audio_list'][0] for data in res_list], labels)
        # for data, score in zip(res_list, scores):
        #     data['model_eval'] = score
        # self.save()

        # # Gemini-2.0
        # query_list = [form_gemini_mm_query(f"Does the given music obviously use the instrument {inst['instrument']}? Explain step "
        #                                    f"by step and end your answer with \"Yes\" or \"No\".", audios=[data['audio_list'][0]])
        #               for data, inst in zip(res_list, inst_list)]
        # responses = batch(query_gemini, query_list, model='gemini-2.5-pro-preview-03-25', temperature=0.0, num_worker=1)
        # for idx, data in enumerate(res_list):
        #     data['model_eval'] = float('yes' in responses[idx].strip().lower()[-20:])
        # self.save()

        # ClapScore audio-audio
        for data, inst in zip(res_list, inst_list):
            if len(data['audio_list']) != 1:
                data['model_eval'] = 0.0
                continue
            ref_audio_list = []
            for i in range(100):
                ref_audio, sr = librosa.load(f'./datasets/openmic-2018/{inst["instrument"]}/{i}.mp3')
                if sr != SAMPLE_RATE:
                    ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                ref_audio_list.append(ref_audio)
            data['model_eval'] = compute_clapscore_aa(data['audio_list'][0], ref_audio_list)
        self.save()

    def evaluate_tempo(self):
        res_list = []
        inst_list = []
        for data, inst in zip(self.res_list, self.inst_list):
            if 'tempo' in inst:
                res_list.append(data)
                inst_list.append(inst)
        if len(inst_list) == 0:
            return

        from beat_this.inference import Audio2Beats
        model = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)
        for data, inst in zip(res_list, inst_list):
            if len(data['audio_list']) != 1:
                data['auto_eval'] = 0.0
                data['auto_eval_score'] = 0.0
                continue
            audio = librosa.effects.trim(data['audio_list'][0])[0]
            beats, _ = model(audio, SAMPLE_RATE)
            bpm = len(beats) * SAMPLE_RATE * 60 / audio.shape[0]
            data['auto_eval'] = bpm
            data['auto_eval_score'] = float(abs(inst['tempo'] - bpm) < 5)
        self.save()

    def evaluate(self):
        self.evaluate_instrument()
        self.evaluate_tempo()

    def human_evaluate(self):
        res_list = []
        inst_list = []
        for data, inst in zip(self.res_list, self.inst_list):
            if 'instrument' in inst:
                res_list.append(data)
                inst_list.append(inst)
        if len(inst_list) == 0:
            return
        instruments = tuple(set(inst['instrument'] for inst in inst_list)) + ('None of the above',)
        back_list = []
        for instrument in instruments[: -1]:
            back_list.append([])
            back_list[-1].append(instrument)
            for i in range(5):
                back_list[-1].append(f'./seed_instruction/audio/{instrument}_{i}.wav')

        human_inst_list = []
        human_data_list = []
        idx = 0
        for data in self.res_list:
            if len(data['audio_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            human_inst_list.append(f'What is the dominant instrument played the given audio?\n'
                            f'Reminder:\n'
                            f'1. Failed generation should be considered as none of the above.\n'
                            f'2. Choose multiple labels only when you are unsure or the given audio can fall into different types.')
            human_data_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = MultiLabelInterface(
            label_list=instruments,
            eval_inst_list=human_inst_list,
            data_list=human_data_list,
            back_list=back_list,
            multi_choice=True,
            mm_type='a'
        )
        interface.start()
        for data, inst in zip(res_list, inst_list):
            if data['human_eval'] != FAILED_TOKEN:
                data['human_eval'] = [instruments[i] for i in interface.eval_list[data['human_eval']]]
            else:
                data['human_eval'] = []
            data['human_eval_score'] = float(inst['instrument'] in data['human_eval'])
        self.save()


class AMusicInstrument(EvalUnit):
    def __init__(self, model_name: str, sample_size=4):
        self.eval_unit = AMusicAttribute(model_name=model_name, sample_size=sample_size)

    @property
    def res_list(self):
        return self.eval_unit.res_list

    def evaluate(self):
        self.eval_unit.evaluate_instrument()

    def human_evaluate(self):
        self.eval_unit.human_evaluate()

    def compute_accuracy(self, threshold=0.62, return_list=False):
        model_eval_list = [data['model_eval'] for data in self.eval_unit.res_list if 'model_eval' in data]
        model_eval_list = [float(model_eval > threshold) for model_eval in model_eval_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self, threshold=0.62):
        model_eval_list = [data['model_eval'] for data in self.eval_unit.res_list if 'model_eval' in data]
        human_eval_list = [data['human_eval_score'] for data in self.eval_unit.res_list if 'human_eval_score' in data]

        # # Optimal threshold
        # threshold = find_optimal_threshold(model_eval_list, human_eval_list)

        model_eval_list = [float(model_eval > threshold) for model_eval in model_eval_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)

    def save(self, save_all=False):
        self.eval_unit.save(save_all=save_all)


class AMusicTempo(EvalUnit):
    def __init__(self, model_name: str, sample_size=4):
        self.eval_unit = AMusicAttribute(model_name=model_name, sample_size=sample_size)

    @property
    def res_list(self):
        return self.eval_unit.res_list

    def evaluate(self):
        self.eval_unit.evaluate_tempo()

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [data['auto_eval_score'] for data in self.eval_unit.res_list if 'auto_eval_score' in data]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)

    def save(self, save_all=False):
        self.eval_unit.save(save_all=save_all)


class AMusicIntensity(EvalUnit):
    inst_name = 'a_music_intensity'

    def evaluate(self):
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['auto_eval'] = 0.0
                continue
            audio = librosa.effects.trim(data['audio_list'][0])[0]
            timestep = min(4, len(audio) // (SAMPLE_RATE * 2))
            if inst['intensity'][0] == 'start':
                audio = audio[: round(timestep * SAMPLE_RATE)]
            else:
                audio = audio[-round(timestep * SAMPLE_RATE):]
            intensity = librosa.feature.rms(y=audio)[0]
            norm_intensity = (intensity - min(intensity)) / (max(intensity) - min(intensity))
            times = librosa.frames_to_time(np.arange(len(norm_intensity)), sr=SAMPLE_RATE)
            peaks = list(find_peaks(norm_intensity, distance=4)[0])
            if peaks[0] >= 5:
                peaks.insert(0, 0)
            if peaks[-1] < len(norm_intensity) - 5:
                peaks.append(len(norm_intensity) - 1)
            slope, _, _, _, stderr = linregress(times[peaks], norm_intensity[peaks])
            trend = 'fade in' if (slope > 0.18 and stderr < 0.04) else \
                ('fade out' if (slope < -0.18 and stderr < 0.04) else FAILED_TOKEN)
            data['auto_eval'] = float(trend == inst['intensity'][1])

            # # Visualize
            # from matplotlib import pyplot as plt
            # plt.plot(times, norm_intensity * 100.0, label='Intensity', alpha=0.6)
            # plt.plot(times[peaks], norm_intensity[peaks] * 100.0, label='Peaks', alpha=0.6)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Intensity (%)')
            # plt.title(f'Normalized slop: {slope:.3f}, Stderr: {stderr:.3f}')
            # plt.legend()
            # plt.show()
        self.save()

    def compute_accuracy(self, return_list=False):
        auto_eval_list = [data['auto_eval'] for data in self.res_list]
        if return_list:
            return auto_eval_list
        return np.mean(auto_eval_list)


class AMusicExclude(EvalUnit):
    inst_name = 'a_music_exclude'

    def evaluate(self):
        # # ClapScore audio-text
        # labels = [inst['instrument'] + ' music' for inst in self.inst_list]
        # scores = compute_clapscore_at([data['audio_list'][0] for data in self.res_list], labels)
        # for data, score in zip(self.res_list, scores):
        #     data['model_eval'] = score
        # self.save()

        # # Gemini-2.0
        # query_list = [form_gemini_mm_query(f"Does the given music obviously use the instrument {inst['instrument']}? Explain step "
        #                                    f"by step and end your answer with \"Yes\" or \"No\".", audios=[data['audio_list'][0]])
        #               for data, inst in zip(self.res_list, self.inst_list)]
        # responses = batch(query_gemini, query_list, model='gemini-2.5-pro-preview-03-25', temperature=0.0, num_worker=1)
        # for idx, data in enumerate(self.res_list):
        #     data['model_eval'] = float('yes' in responses[idx].strip().lower()[-20:])
        # self.save()

        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['model_eval'] = 1.0
                continue
            ref_audio_list = []
            for i in range(100):
                ref_audio, sr = librosa.load(f'./datasets/openmic-2018/{inst["instrument"]}/{i}.mp3')
                if sr != SAMPLE_RATE:
                    ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                ref_audio_list.append(ref_audio)
            data['model_eval'] = compute_clapscore_aa(data['audio_list'][0], ref_audio_list)
        self.save()

    def human_evaluate(self):
        instruments = tuple(set(inst['instrument'] for inst in self.inst_list)) + ('None of the above',)
        back_list = []
        for instrument in instruments[: -1]:
            back_list.append([])
            back_list[-1].append(instrument)
            for i in range(5):
                back_list[-1].append(f'./seed_instruction/audio/{instrument}_{i}.wav')
        human_inst_list = []
        human_data_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
            if len(data['audio_list']) != 1:
                data['human_eval'] = FAILED_TOKEN
                continue
            human_inst_list.append(f"Does {inst['instrument']} exist in the given music?")
            human_data_list.append(data)
            data['human_eval'] = idx
            idx += 1
        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=human_inst_list,
            data_list=human_data_list,
            back_list=back_list,
            mm_type='a'
        )
        interface.start()
        for data in self.res_list:
            data['human_eval'] = float(interface.eval_list[data['human_eval']]) if 'human_eval' != FAILED_TOKEN else 0.0
        self.save()

    def compute_accuracy(self, threshold=0.62, return_list=False):
        model_eval_list = [data['model_eval'] for data in self.res_list]
        model_eval_list = [float(model_eval < threshold) for model_eval in model_eval_list]
        if return_list:
            return model_eval_list
        return np.mean(model_eval_list)

    def compute_correlation(self, threshold=0.62):
        model_eval_list = [data['model_eval'] for data in self.res_list]
        human_eval_list = [data['human_eval'] for data in self.res_list]

        # # Optimal threshold
        # threshold = find_optimal_threshold([1.0 - m for m in model_eval_list], human_eval_list)

        model_eval_list = [float(model_eval < threshold) for model_eval in model_eval_list]
        return calculate_agreement(model_eval_list, human_eval_list), calculate_pearson(model_eval_list, human_eval_list)


if __name__ == '__main__':
    pass
