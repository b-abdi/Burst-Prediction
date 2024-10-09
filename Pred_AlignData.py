from scipy import signal
from scipy.io import loadmat
import numpy as np
import os
import tensorflow as tf
from scipy import interpolate

# Set only the second GPU to be visible to TensorFlow
devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {devices}")
tf.config.set_visible_devices(devices[1], 'GPU')
tf.config.experimental.set_memory_growth(devices[1], True)


def burst_threshod_avg(lfp, percentile=0.75):
    range_lfp = np.arange(len(lfp))
    abs_lfp = np.abs(lfp)
    peaks = signal.find_peaks(abs_lfp)[0]
    interPolate = interpolate.interp1d(peaks, abs_lfp[peaks], bounds_error=False, fill_value="extrapolate")
    intPol_lfp = interPolate(range_lfp)
    sort_lfp = np.sort(intPol_lfp)
    ind_threshold = int(percentile * len(sort_lfp))
    threshold = sort_lfp[ind_threshold]
    return threshold


def burst_annotation(lfp, fs_new, thrshl, diff_bet_burst, long_burst_thr):
    range_lfp = np.arange(len(lfp))
    abs_lfp = np.abs(lfp)
    peaks = signal.find_peaks(abs_lfp)[0]
    interPolate = interpolate.interp1d(peaks, abs_lfp[peaks], bounds_error=False, fill_value="extrapolate")
    intPol_lfp = interPolate(range_lfp)

    diff_bet_burst_sam = int(diff_bet_burst * fs_new)
    ind_st = int(0.2 * fs_new)  # start after 200 ms
    burst_st = []
    burst_end = []
    ind_end_all = []
    ind_st_all = []
    while ind_st < len(intPol_lfp):
        amp = intPol_lfp[ind_st]
        if amp > thrshl:
            ind_st_all.append(ind_st)
            ind_end = np.argwhere(intPol_lfp[ind_st: ind_st + 10 * fs_new] < thrshl)
            if len(ind_end) > 1:
                ind_end_all.append(ind_st + ind_end[0])
            else:
                break
            # IF the difference between two burst is less than a threshold, those are considered as one burst
            if len(burst_st) >= 2:
                if (ind_st - ind_end_all[-2]) < diff_bet_burst_sam:
                    if (ind_end_all[-2] - ind_st_all[-2]) > (long_burst_thr * fs_new):
                        del burst_st[-1]
                        del burst_end[-1]
                        del ind_st_all[-1]
                        del ind_end_all[-2]
                    else:
                        del ind_st_all[-1]
                        del ind_end_all[-2]

            if (ind_end_all[-1] - ind_st_all[-1]) > (long_burst_thr * fs_new):
                burst_st.append(ind_st_all[-1] / fs_new)
                burst_end.append(ind_end_all[-1] / fs_new)

            ind_st = ind_st + int(ind_end[0]) - 1
        ind_st = ind_st + 2

    return burst_st, burst_end


# Extracting beta bursts
def data_annotation(lfp, burst_st, burst_end, fs, prior_time_burst, prior_time_nonburst, segment_len, stride_nonburst,
                    training=False):
    seg_len_sam = int(segment_len * fs)
    prior_burst_sam = int(prior_time_burst * fs)
    stride_nonburst_sam = int(stride_nonburst * fs)

    # ignor the first burst since the begining of signal may include burst
    burst = []
    nonburst = []
    for i in range(1, len(burst_st)):
        onset = burst_st[i]
        st_seg = int(onset * fs) - prior_burst_sam - seg_len_sam
        end_seg = int(onset * fs) - prior_burst_sam
        if training:
            burst.append(lfp[st_seg - 3: end_seg - 3])
            burst.append(lfp[st_seg:end_seg])
            burst.append(lfp[st_seg + 3: end_seg + 3])
        else:
            burst.append(lfp[st_seg:end_seg])

        st_nonburst = int((burst_end[i - 1] + stride_nonburst) * fs)
        end_nonburst = int((burst_st[i] - prior_time_nonburst) * fs)
        len_nonburst_sam = end_nonburst - st_nonburst
        if len_nonburst_sam > seg_len_sam:
            for st in np.arange(st_nonburst, end_nonburst - seg_len_sam, stride_nonburst_sam):
                nonburst.append(lfp[st:st + seg_len_sam])

    return burst, nonburst


Subject_Name = ['LN01', 'LN03', 'LN06', 'LN07', 'LN08', 'LN09', 'DP', 'JN', 'DF', 'JA', 'JP', 'LN11', 'LN_C10',
                'LN_C20', 'LN_C21', 'DS', 'LN02']

hemisphere = 'Right'
for sub_name in Subject_Name:
    data_path = '/mnt/Oswal_Lab/UCL_LFP_selected/'
    data_path_sub = os.path.join(data_path, sub_name, hemisphere)
    pred_time = [0, 20, 40, 60, 80, 100]
    for pred_t in pred_time:
        data_dir = os.listdir(data_path_sub)
        if '.DS_Store' in data_dir:
            data_dir.remove('.DS_Store')
        num_session = len(data_dir)
        session = 0
        for lisDir in data_dir:
            session = session + 1

            data_path_trial = os.path.join(data_path_sub, lisDir)
            data = loadmat(data_path_trial)
            print('Keys: %s' % data.keys())
            if hemisphere == 'Left':
                datadata = data['DataLeft']
            elif hemisphere == 'Right':
                datadata = data['DataRight']
            else:
                raise TypeError('There is an error. The hemisphere is not defined.')

            print(datadata.dtype)
            label = datadata['Label'][0][0][:, 0]
            sig = datadata['Signal']
            sig = sig[0, 0]
            fs = datadata['SamplingRate'][0][0][0]
            freq_beta_peak = datadata['beta_freq'][0][0][0]
            channel_beta_peak = datadata['beta_ch'][0, 0]
            print('Length data is: ', (sig.shape[-1] / fs)[0])

            ### Preprocessing signal
            lfp_ind = []
            beta_ch = []
            for i, ch_label in enumerate(label):
                if ch_label == channel_beta_peak:
                    beta_ch.append(i)
            LFP = sig[np.array(beta_ch), :]

            # Narrow filter
            fl_left = freq_beta_peak - 3
            fh_left = freq_beta_peak + 3
            cut_off_freq = [fl_left, fh_left]
            sos = signal.butter(6, cut_off_freq, 'bandpass', fs=fs, output='sos')
            LFP_left_filt = signal.sosfilt(sos, LFP, axis=-1)

            # Downsample and select beta channel
            fs_new = 600
            sampling_ind = np.arange(0, LFP_left_filt.shape[-1], int(fs / fs_new))
            LFP_downsam = LFP_left_filt[0, sampling_ind]

            # Test And Train
            percentile = 0.75
            win_len = 0.2
            long_burst_thr = 0
            diff_bet_burst = 0
            prior_time_burst_ms = pred_t
            prior_time_burst = prior_time_burst_ms / 1000
            prior_time_nonburst = 0.15
            stride_nonburst = 0.05 #50 ms
            if num_session == 1:
                test_per = 0.15
                val_per = 0.15
                LFP_train = LFP_downsam[0:-int((test_per + val_per) * LFP_downsam.shape[-1])]
                burst_thr = burst_threshod_avg(LFP_train, percentile)
                burst_st_tr, burst_end_tr = burst_annotation(LFP_train, fs_new, burst_thr, diff_bet_burst,
                                                             long_burst_thr)
                burst_tr, nonburst_tr = data_annotation(LFP_train, burst_st_tr, burst_end_tr, fs_new, prior_time_burst,
                                                        prior_time_nonburst, win_len, stride_nonburst, training=True)

                LFP_val = LFP_downsam[
                          -int((test_per + val_per) * LFP_downsam.shape[-1]):-int(test_per * LFP_downsam.shape[-1])]
                burst_st_val, burst_end_val = burst_annotation(LFP_val, fs_new, burst_thr, diff_bet_burst,
                                                               long_burst_thr)
                burst_val, nonburst_val = data_annotation(LFP_val, burst_st_val, burst_end_val, fs_new,
                                                          prior_time_burst,
                                                          prior_time_nonburst, win_len, stride_nonburst, training=False)

                LFP_test = LFP_downsam[-int(test_per * LFP_downsam.shape[-1]):]
                burst_st_test, burst_end_test = burst_annotation(LFP_test, fs_new, burst_thr, diff_bet_burst,
                                                                 long_burst_thr)
                burst_test, nonburst_test = data_annotation(LFP_test, burst_st_test, burst_end_test, fs_new,
                                                            prior_time_burst,
                                                            prior_time_nonburst, win_len, stride_nonburst,
                                                            training=False)

            else:
                if session == 1:
                    LFP_train = LFP_downsam
                    burst_thr = burst_threshod_avg(LFP_train, percentile)
                    burst_st_tr, burst_end_tr = burst_annotation(LFP_train, fs_new, burst_thr, diff_bet_burst,
                                                                 long_burst_thr)
                    burst_tr, nonburst_tr = data_annotation(LFP_train, burst_st_tr, burst_end_tr, fs_new,
                                                            prior_time_burst,
                                                            prior_time_nonburst, win_len, stride_nonburst,
                                                            training=True)

                else:
                    test_per = 0.5
                    LFP_val = LFP_downsam[0: -int(test_per * LFP_downsam.shape[-1])]
                    burst_st_val, burst_end_val = burst_annotation(LFP_val, fs_new, burst_thr, diff_bet_burst,
                                                                   long_burst_thr)
                    burst_val, nonburst_val = data_annotation(LFP_val, burst_st_val, burst_end_val, fs_new,
                                                              prior_time_burst,
                                                              prior_time_nonburst, win_len, stride_nonburst,
                                                              training=False)

                    LFP_test = LFP_downsam[-int(test_per * LFP_downsam.shape[-1]):]
                    burst_st_test, burst_end_test = burst_annotation(LFP_test, fs_new, burst_thr, diff_bet_burst,
                                                                     long_burst_thr)
                    burst_test, nonburst_test = data_annotation(LFP_test, burst_st_test, burst_end_test, fs_new,
                                                                prior_time_burst,
                                                                prior_time_nonburst, win_len, stride_nonburst,
                                                                training=False)

        burst_tr_arr = np.array(burst_tr)
        burst_tr_arr = np.expand_dims(burst_tr_arr, -1)

        nonburst_tr_arr = np.array(nonburst_tr)
        nonburst_tr_arr = np.expand_dims(nonburst_tr_arr, -1)

        burst_val_arr = np.array(burst_val)
        burst_val_arr = np.expand_dims(burst_val_arr, -1)

        nonburst_val_arr = np.array(nonburst_val)
        nonburst_val_arr = np.expand_dims(nonburst_val_arr, -1)

        burst_test_arr = np.array(burst_test)
        burst_test_arr = np.expand_dims(burst_test_arr, -1)

        nonburst_test_arr = np.array(nonburst_test)
        nonburst_test_arr = np.expand_dims(nonburst_test_arr, -1)

        # Randomizing the datasets
        # The number of nonburst segments is selected to be the same number as burst segments
        num_burst_tr = len(burst_tr_arr)
        rand_ind_tr_nonburst = np.arange(len(nonburst_tr_arr))
        np.random.seed(101)
        np.random.shuffle(rand_ind_tr_nonburst)
        nonburst_tr_arr_sel = nonburst_tr_arr[rand_ind_tr_nonburst[:num_burst_tr], :, :]
        train_data = np.concatenate([nonburst_tr_arr_sel, burst_tr_arr], axis=0)
        train_label = np.concatenate([np.zeros(len(nonburst_tr_arr_sel)), np.ones(len(burst_tr_arr))])
        rand_ind_train = np.arange(len(train_data))
        np.random.seed(101)
        np.random.shuffle(rand_ind_train)
        train_label = train_label[rand_ind_train]
        train_data = train_data[rand_ind_train, :, :]

        num_burst_val = len(burst_val_arr)
        rand_ind_val_nonburst = np.arange(len(nonburst_val_arr))
        np.random.seed(101)
        np.random.shuffle(rand_ind_val_nonburst)
        nonburst_val_arr_sel = nonburst_val_arr[rand_ind_val_nonburst[:num_burst_val], :, :]
        val_data = np.concatenate([nonburst_val_arr_sel, burst_val_arr], axis=0)
        val_label = np.concatenate([np.zeros(len(nonburst_val_arr_sel)), np.ones(len(burst_val_arr))])
        rand_ind_val = np.arange(len(val_data))
        np.random.seed(101)
        np.random.shuffle(rand_ind_val)
        val_label = val_label[rand_ind_val]
        val_data = val_data[rand_ind_val, :, :]

        num_burst_test = len(burst_test_arr)
        rand_ind_test_nonburst = np.arange(len(nonburst_test_arr))
        np.random.seed(101)
        np.random.shuffle(rand_ind_test_nonburst)
        nonburst_test_arr_sel = nonburst_test_arr[rand_ind_test_nonburst[:num_burst_test], :, :]
        test_data = np.concatenate([nonburst_test_arr_sel, burst_test_arr], axis=0)
        test_label = np.concatenate([np.zeros(len(nonburst_test_arr_sel)), np.ones(len(burst_test_arr))])
        rand_ind_test = np.arange(len(test_data))
        np.random.seed(101)
        np.random.shuffle(rand_ind_test)
        test_label = test_label[rand_ind_test]
        test_data = test_data[rand_ind_test, :, :]

        # Model CNN
        num_filt = 128
        kernel = 5
        n_dense = 512
        drop_rate = 0.5
        act_layer = 'relu'
        init_rate = 0.0001
        opt = tf.keras.optimizers.Adam(init_rate)
        num_batch = 8
        num_epoch = 50

        in_shape = train_data.shape[1:]
        in_layer = tf.keras.layers.Input(shape=in_shape)
        conv_1 = tf.keras.layers.Conv1D(num_filt, kernel_size=kernel, strides=1, padding='same')(in_layer)
        act_1 = tf.keras.layers.Activation('relu')(conv_1)
        down_1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(act_1)
        conv_2 = tf.keras.layers.Conv1D(num_filt * 2, kernel_size=kernel, strides=1, activation=act_layer,
                                        padding='same')(down_1)
        act_2 = tf.keras.layers.Activation('relu')(conv_2)
        down_2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(act_2)
        conv_3 = tf.keras.layers.Conv1D(num_filt * 4, kernel_size=kernel, strides=1, activation=act_layer,
                                        padding='same')(down_2)
        act_3 = tf.keras.layers.Activation('relu')(conv_3)
        down_3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(act_3)
        flatten = tf.keras.layers.Flatten()(down_3)
        dense = tf.keras.layers.Dense(n_dense, activation=act_layer)(flatten)
        dense = tf.keras.layers.Dropout(drop_rate)(dense)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
        model = tf.keras.models.Model(in_layer, out)

        model.summary()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
        stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

        decay = 0.01

        def lr_time_based_decay(epoch, lr):
            lr_rate = lr / (1 + (decay * epoch))
            return lr_rate

        Lr_callbak = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay)

        model.fit(train_data, train_label,
                  batch_size=num_batch,
                  epochs=num_epoch,
                  validation_data=(val_data, val_label),
                  callbacks=[stop_callback, Lr_callbak])

        # Evaluation of the validation data
        tr = 0.5
        val_prob = model.predict(val_data_all)
        pred = np.where(val_prob > tr, 1, 0).ravel()
        tn, fp, fn, tp = confusion_matrix(val_label_all, pred).ravel()
        sen = tp / (tp + fn)
        prc = tp / (tp + fp)
        spc = tn / (tn + fp)
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision, recall, thrshld_pre_rec = precision_recall_curve(val_label_all, val_prob)
        pr_auc = auc(recall, precision)
        print('Validation Performance for ', pred_t, ' ms: SEN: ', sen, ', PRC: ', prc, ', SPC: ', spc, ', ACC: ', acc,
              'PR_AUC: ', pr_auc)

        # Evaluation the test data
        test_prob = model.predict(test_data_all)
        pred = np.where(test_prob > tr, 1, 0).ravel()
        tn, fp, fn, tp = confusion_matrix(test_label_all, pred).ravel()
        sen_test = tp / (tp + fn)
        prc_test = tp / (tp + fp)
        spc_test = tn / (tn + fp)
        acc_test = (tp + tn) / (tp + tn + fp + fn)
        precision, recall, thrshld_pre_rec = precision_recall_curve(test_label_all, test_prob)
        pr_auc = auc(recall, precision)

        print('Test Dataset Performance: SEN: ', sen_test, ', PRC: ', prc_test, ', SPC: ', spc_test, ', ACC: ',
              acc_test, 'PR_AUC: ', pr_auc)

        Performance[sub_name] = {'Pred_time': pred_t, 'prob_val': val_prob, 'prob_test': test_prob,
                                 'data_val': val_data_all, 'data_test': test_data_all,
                                 'label_val': val_label_all, 'label_test': test_label_all}

