import itertools
import os

import matplotlib.pyplot as plt

from eval import *


class ASound(EvalUnit):
    inst_name = 'a_sound'

    @abstractmethod
    def _evaluate(self):
        pass

    def evaluate(self):
        audio_list, label_list, human_eval_res_list, idx_list = self._evaluate()

        # CLAPScore audio-text
        scores = compute_clapscore_at(audio_list, label_list)
        for idx, data in enumerate(self.res_list):
            data['clapscore_at'] = [scores[i] if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        self.save()

        # CLAPScore audio-audio
        ref_audio_map = pd.read_csv('./datasets/ESC-50/dataset.csv')
        scores = []
        for audio, label in zip(audio_list, label_list):
            file_list = ref_audio_map[ref_audio_map['category'] == label]['filename'].tolist()
            ref_audio_list = []
            for file_dir in file_list:
                ref_audio, sr = librosa.load('./dataset/ESC-50/audio/' + file_dir)
                if sr != SAMPLE_RATE:
                    ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                ref_audio_list.append(ref_audio)
            scores.append(compute_clapscore_aa(audio, ref_audio_list))
        for idx, data in enumerate(self.res_list):
            data['clapscore_aa'] = [scores[i] if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        self.save()

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

        # Gemini-2.0
        query_list = [form_gemini_mm_query(f"Is the given audio the sound of {l}? Answer only yes or no.", audios=[a])
                      for a, l in zip(audio_list, label_list)]
        responses = batch(query_gemini, query_list, model='gemini-2.0-flash-exp', temperature=0.0, num_worker=1)
        for idx, data in enumerate(self.res_list):
            data['gemini_eval'] = [float('yes' in responses[i].lower()) if i != FAILED_TOKEN else 0.0 for i in idx_list[idx]]
        self.save()

    def calculate_metrics(self, method='clapscore_aa', threshold=0.6):
        model_eval_list = [res[method] for res in self.res_list]
        human_eval_list = [res['human_eval'] for res in self.res_list]

        # Optimal threshold
        model_eval_cat = list(itertools.chain(*model_eval_list))
        human_eval_cat = list(itertools.chain(*human_eval_list))
        threshold = find_optimal_threshold(model_eval_cat, human_eval_cat)

        model_eval_list = [np.mean([e > threshold for e in model_eval]) for model_eval in model_eval_list]
        human_eval_list = [np.mean(human_eval) for human_eval in human_eval_list]

        print(f"Model evaluation accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluation accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))
        print(f"Agreement for {self.inst_name}: ", calculate_agreement(model_eval_list, human_eval_list))


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
                audio_list.append(data['audio_list'][0][: SAMPLE_RATE * 2])
                label_list.append(inst['start'])
                human_eval_res_list.append({'query': data['query'], 'audio_list': [audio_list[-1]]})
                idx_list[-1].append(idx)
                idx += 1
            if 'end' in inst:
                audio_list.append(data['audio_list'][0][-SAMPLE_RATE * 2:])
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
        return (
            [data['audio_list'][0] for data in self.res_list],
            [inst['target'] for inst in self.inst_list],
            self.res_list,
            [[i] for i in range(len(self.res_list))]
        )


class ASoundSilence(ASound):
    inst_name = 'a_sound_silence'

    def _evaluate(self):
        audio_list = []
        label_list = []
        human_eval_res_list = []
        idx_list = []
        idx = 0
        for data, inst in zip(self.res_list, self.inst_list):
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

    def evaluate(self, attribute_list=('gender', 'pitch', 'pitch', 'speed')):
        transcripts, wers = transcribe_speech(
            [data['audio_list'][0] for data in self.res_list],
            [inst['text'] for inst in self.inst_list],
            self.language
        )
        for trans, wer, data in zip(transcripts, wers, self.res_list):
            data['transcript'] = trans
            data['wer'] = wer
        self.save()

        from libs.SpeechGenderCls import get_gender
        genders = get_gender([f'./output/{self.model_name}/audio/{self.inst_name}_{i}.wav' for i in range(len(self.res_list))])
        for data, inst, gender in zip( self.res_list, self.inst_list, genders):
            if 'model_eval' in data and len(data['model_eval']) == 4:
                data['model_eval'][0] = gender
                data['model_eval_score'][0] = float(gender == ('male', 'female').index(inst['gender']))
            else:
                data['model_eval'] = [gender, FAILED_TOKEN, FAILED_TOKEN, FAILED_TOKEN]
                data['model_eval_score'] = [float(gender == ('male', 'female').index(inst['gender'])), FAILED_TOKEN, FAILED_TOKEN, FAILED_TOKEN]
        self.save()

        for data, inst in zip(self.res_list, self.inst_list):
            analysis, analysis_score = calculate_pitch_and_speed(
                data['audio_list'][0], data['model_eval'][0], data['transcript'], inst, self.language
            )
            data['model_eval'][1:] = analysis
            data['model_eval_score'][1:] = analysis_score
        self.save()

        option_list = (('male', 'female'), ('low', 'medium', 'high'), ('low', 'medium', 'high'), ('low', 'high'))
        question_list = (
            'Is the given speech in male or female voice? Please compare between difference speeches to have a better calibration.',
            'Is the given speech in low or high pitch? Please compare between difference speeches to have a better calibration.',
            'Is the given speech in low or high pitch? Please compare between difference speeches to have a better calibration.',
            'Is the given speech in low or high speaking rate? Please compare between difference speeches to have a better calibration.',
        )
        for data in self.res_list:
            data['human_eval'] = [data['human_eval'][0], FAILED_TOKEN, FAILED_TOKEN, data['human_eval'][3]]
            data['human_eval_score'] = [data['human_eval_score'][0], FAILED_TOKEN, FAILED_TOKEN, data['human_eval_score'][3]]
        for i in range(len(attribute_list)):
            if attribute_list[i] == FAILED_TOKEN:
                continue
            idx = 0
            data_list = []
            for data, inst in zip(self.res_list, self.inst_list):
                if attribute_list[i] in inst:
                    if attribute_list[i] == 'pitch' and data['human_eval'][0] != i - 1:
                        continue
                    data['human_eval'][i] = idx
                    data['human_eval_score'][i] = idx
                    data_list.append(data['audio_list'][0])
                    idx += 1
                else:
                    data['human_eval'][i] = FAILED_TOKEN
                    data['human_eval_score'][i] = FAILED_TOKEN
            interface = CalibratedLabelInterface(
                label_list=option_list[i],
                eval_inst=question_list[i],
                data_list=data_list,
            )
            interface.start()
            for data, inst in zip(self.res_list, self.inst_list):
                if data['human_eval'][i] != FAILED_TOKEN:
                    if interface.eval_list[data['human_eval'][i]] == FAILED_TOKEN:
                        data['human_eval'][i] = FAILED_TOKEN
                        data['human_eval_score'][i] = FAILED_TOKEN
                    else:
                        data['human_eval'][i] = interface.eval_list[data['human_eval'][i]]
                        data['human_eval_score'][i] = float(option_list[i][data['human_eval'][i]] in (inst[attribute_list[i]], 'medium'))
            self.save()

    def calculate_metrics(self, find_threshold=False):
        print(f'Word Error Rate for {self.inst_name}: ', np.mean([data['wer'] for data in self.res_list]))
        option_list = [('male', 'female'), ('low', 'medium', 'high'), ('low', 'medium', 'high'), ('low', 'high')]
        for i, task in enumerate(('gender', 'pitch', 'pitch', 'speed')):
            model_eval_list = []
            label_list = []
            target_list = []
            human_eval_list = []
            for data, inst in zip(self.res_list, self.inst_list):
                if data['human_eval'][i] != FAILED_TOKEN:
                    if task == 'pitch' and (data['human_eval'][0] != i - 1 or data['model_eval_score'][i] == FAILED_TOKEN):
                        continue
                    if find_threshold:
                        model_eval_list.append(data['model_eval'][i])
                    else:
                        model_eval_list.append(data['model_eval_score'][i])
                    human_eval_list.append(data['human_eval_score'][i])
                    target_list.append(inst[task])
                    label_list.append(data['human_eval'][i])
            # plt.hist(model_eval_list, bins=10, edgecolor='black', alpha=0.7)
            # plt.show()
            if find_threshold:
                if task != 'pitch':
                    thres = find_optimal_threshold(model_eval_list, label_list)
                    model_eval_list = [int(me > thres) for me in model_eval_list]
                    model_eval_list = [float(option_list[i][me] == tar) for me, tar in zip(model_eval_list, target_list)]
                else:
                    low, high = find_optimal_thresholds(model_eval_list, label_list)
                    model_eval_list = [0 if me < low else (2 if me > high else 1) for me in model_eval_list]
                    model_eval_list = [float(option_list[i][me] in (tar, 'medium')) for me, tar in zip(model_eval_list, target_list)]

            print(f"Model evaluated {task} accuracy for {self.inst_name}: ", np.mean(model_eval_list))
            print(f"Human evaluated {task} accuracy for {self.inst_name}: ", np.mean(human_eval_list))
            print(f"Pearson Correlation of {task} for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))
            print(f"Agreement of {task} for {self.inst_name}: ", calculate_agreement(model_eval_list, human_eval_list))

        model_eval_list = [np.min([me for me in data['model_eval_score'] if me != FAILED_TOKEN]) for data in self.res_list]
        human_eval_list = [np.min([me for me in data['human_eval_score'] if me != FAILED_TOKEN]) for data in self.res_list]
        print(f"Model evaluated accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluated accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(f"Pearson Correlation of for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))
        print(f"Agreement of for {self.inst_name}: ", calculate_agreement(model_eval_list, human_eval_list))


class ASpeechChinese(ASpeechAttribute):
    inst_name = 'a_speech_chinese'
    language = 'chinese'


class ASpeechImitate(EvalUnit):
    inst_name = 'a_speech_imitate'

    def evaluate(self):
        transcripts, wers = transcribe_speech(
            [data['audio_list'][0] for data in self.res_list],
            [inst['text'] for inst in self.inst_list],
        )
        for trans, wer, data in zip(transcripts, wers, self.res_list):
            data['transcript'] = trans
            data['wer'] = wer
        self.save()

        self.load_inst_mm()
        audio_list = [data['audio_list'][0] for data in self.res_list]
        ref_audio_list = [inst['audio_list'][0] for inst in self.inst_list]
        sim_scores = calculate_speech_similarity(audio_list, ref_audio_list)
        for data, sim in zip(self.res_list, sim_scores):
            data['model_eval'] = sim
        self.save()

        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=['Are the speeches coming from the same speaker?'] * len(self.res_list),
            data_list=self.res_list,
            ref_list=ref_audio_list,
            mm_type='a'
        )
        interface.start()
        for data, human_eval in zip(self.res_list, interface.eval_list):
            data['human_eval'] = 1.0 - human_eval
        self.save()

    def calculate_metrics(self, threshold=0.865):
        print(f'Word Error Rate for {self.inst_name}: ', np.mean([data['wer'] for data in self.res_list]))
        model_eval_list = [data['model_eval'] for data in self.res_list]
        human_eval_list = [data['human_eval'] for data in self.res_list]
        model_eval_list = [model_eval > threshold for model_eval in model_eval_list]

        print(f"Model evaluated accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluated accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(f"Pearson Correlation of for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))
        print(f"Agreement of for {self.inst_name}: ", calculate_agreement(model_eval_list, human_eval_list))


class ASpeechModify(EvalUnit):
    inst_name = 'a_speech_modify'

    def evaluate(self):
        transcripts, _ = transcribe_speech([data['audio_list'][0] for data in self.res_list])
        for trans, data in zip(transcripts, self.res_list):
            data['transcript'] = trans
        self.save()

        scores = text_instruction_following_verify(
            [data['transcript'] for data in self.res_list],
            [inst['constraint'] for inst in self.inst_list]
        )
        for score, data in zip(scores, self.res_list):
            data['auto_eval'] = score
        self.save()

    def calculate_metrics(self):
        auto_eval_list = [res['auto_eval'] for res in self.res_list]
        print(f"Auto evaluation accuracy for {self.inst_name}: ", np.mean(auto_eval_list))


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
        # from libs.EfficientAT.ex_openmic import inference
        # scores = inference([data['audio_list'][0] for data in self.res_list], [inst['instrument'] for inst in self.inst_list])
        # for data, score in zip(self.res_list, scores):
        #     data['model_eval'][1] = score
        # self.save()

        # labels = [inst['instrument'] + ' music' for inst in self.inst_list]
        # scores = compute_clapscore_at([data['audio_list'][0] for data in self.res_list], labels)
        # for data, score in zip(self.res_list, scores):
        #     data['model_eval'][1]= score
        # self.save()

        for data, inst in zip(res_list, inst_list):
            ref_audio_list = []
            for i in range(100):
                ref_audio, sr = librosa.load(f'./datasets/openmic-2018/{inst["instrument"]}/{i}.mp3')
                if sr != SAMPLE_RATE:
                    ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                ref_audio_list.append(ref_audio)
            data['model_eval'] = compute_clapscore_aa(data['audio_list'][0], ref_audio_list)
            data['model_eval_score'] = float(data['model_eval'] > 0.58)
        self.save()

        instruments = tuple(set(inst['instrument'] for inst in inst_list)) + ('None of the above',)
        back_list = []
        for instrument in instruments[: -1]:
            back_list.append([])
            back_list[-1].append(instrument)
            for i in range(5):
                back_list[-1].append(f'./seed_instruction/audio/{instrument}_{i}.wav')

        interface = MultiLabelInterface(
            label_list=instruments,
            eval_inst_list=[f'What is the dominant instrument played the given audio?\n'
                            f'Reminder:\n'
                            f'1. Failed generation should be considered as none of the above.\n'
                            f'2. Choose multiple labels only when you are unsure or the given audio can fall into different types.'] * len(self.res_list),
            data_list=res_list,
            back_list=back_list,
            shuffle=True,
            multi_choice=True,
            mm_type='a'
        )
        interface.start()
        for data, inst, human_eval in zip(res_list, inst_list, interface.eval_list):
            data['human_eval'] = [instruments[he] for he in human_eval]
            data['human_eval_score'] = float(inst['instrument'] in data['human_eval'])
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
            audio = librosa.effects.trim(data['audio_list'][0])[0]
            beats, _ = model(audio, SAMPLE_RATE)
            bpm = len(beats) * SAMPLE_RATE * 60 / audio.shape[0]
            data['auto_eval'] = bpm
            data['auto_eval_score'] = float(abs(inst['tempo'] - bpm) < 5)
        self.save()

    def evaluate(self, attribute_list=('genre', 'instrument')):
        self.evaluate_instrument()
        self.evaluate_tempo()

    def calculate_metrics(self):
        model_eval_list = [data['model_eval_score'] for data in self.res_list if 'model_eval_score' in data]
        human_eval_list = [data['human_eval_score'] for data in self.res_list if 'human_eval_score' in data]

        print(f"Model evaluated accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluated accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))
        print(f"Agreement for {self.inst_name}: ", calculate_agreement(model_eval_list, human_eval_list))
        auto_eval_list = [data['auto_eval_score'] for data in self.res_list if 'auto_eval_score' in data]
        print(f"Auto evaluated accuracy for {self.inst_name}: ", np.mean(auto_eval_list))


class AMusicIntensity(EvalUnit):
    inst_name = 'a_music_intensity'

    def evaluate(self):
        from scipy.signal import find_peaks
        from scipy.stats import linregress
        timestep = 3.0
        distance = 4

        for data, inst in zip(self.res_list, self.inst_list):
            audio = librosa.effects.trim(data['audio_list'][0])[0]
            audio = audio[round(0.1 * SAMPLE_RATE): -round(0.1 * SAMPLE_RATE)]
            if inst['intensity'][0] == 'start':
                audio = audio[: round(timestep * SAMPLE_RATE)]
            else:
                audio = audio[-round(timestep * SAMPLE_RATE):]
            intensity = librosa.feature.rms(y=audio)[0]
            norm_intensity = (intensity - min(intensity)) / (max(intensity) - min(intensity))
            times = librosa.frames_to_time(np.arange(len(norm_intensity)), sr=SAMPLE_RATE)
            peaks = find_peaks(norm_intensity, distance=distance)[0]
            slope, _, _, _, stderr = linregress(times[peaks], norm_intensity[peaks])
            trend = 'fade in' if (slope > 0.19 and stderr < 0.8) else \
                ('fade out' if (slope < -0.19 and stderr < 0.8) else FAILED_TOKEN)
            data['model_eval'][3] = slope
            data['model_eval_score'][3] = float(trend == inst['intensity'][1])

            # Visualize
            # plt.plot(times, norm_intensity * 100.0, label='Intensity', alpha=0.6)
            # plt.plot(times[peaks], norm_intensity[peaks] * 100.0, label='Peaks', alpha=0.6)
            # plt.xlabel('Time (s)')
            # plt.ylabel('Intensity (%)')
            # plt.title(f'Normalized slop: {slope:.3f}, Stderr: {stderr:.3f}')
            # plt.legend()
            # plt.show()
        self.save()

    def calculate_metrics(self):
        auto_eval_list = [data['model_eval_score'] for data in self.res_list]
        print(f"Auto evaluated accuracy for {self.inst_name}: ", np.mean(auto_eval_list))


class AMusicExclude(EvalUnit):
    inst_name = 'a_music_exclude'

    def evaluate(self):
        for data, inst in zip(self.res_list, self.inst_list):
            ref_audio_list = []
            for i in range(100):
                ref_audio, sr = librosa.load(f'./datasets/openmic-2018/{inst["instrument"]}/{i}.mp3')
                if sr != SAMPLE_RATE:
                    ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                ref_audio_list.append(ref_audio)
            data['model_eval'] = compute_clapscore_aa(data['audio_list'][0], ref_audio_list)
            data['model_eval_score'] = float(data['model_eval'] < 0.585)
        self.save()

        instruments = tuple(set(inst['instrument'] for inst in self.inst_list)) + ('None of the above',)
        back_list = []
        for instrument in instruments[: -1]:
            back_list.append([])
            back_list[-1].append(instrument)
            for i in range(5):
                back_list[-1].append(f'./seed_instruction/audio/{instrument}_{i}.wav')
        interface = MultiLabelInterface(
            label_list=('Yes', 'No'),
            eval_inst_list=[f"Does {inst['instrument']} exist in the given music?" for inst in self.inst_list],
            data_list=self.res_list,
            back_list=back_list,
            shuffle=True,
            mm_type='a'
        )
        interface.start()
        for data, human_eval in zip(self.res_list, interface.eval_list):
            data['human_eval'] = float(human_eval)
        self.save()

    def calculate_metrics(self):
        model_eval_list = [data['model_eval_score'] for data in self.res_list]
        human_eval_list = [data['human_eval'] for data in self.res_list]
        print(f"Model evaluated accuracy for {self.inst_name}: ", np.mean(model_eval_list))
        print(f"Human evaluated accuracy for {self.inst_name}: ", np.mean(human_eval_list))
        print(f"Pearson Correlation for {self.inst_name}: ", calculate_pearson(model_eval_list, human_eval_list))
        print(f"Agreement for {self.inst_name}: ", calculate_agreement(model_eval_list, human_eval_list))


class AMusicLyrics(EvalUnit):
    # TODO
    inst_name = 'a_music_lyrics'

    def evaluate(self):
        pass

    def calculate_metrics(self):
        pass


if __name__ == '__main__':
    a = AMusicAttribute(model_name='MusicGen')
    a.evaluate()
    a.calculate_metrics()
