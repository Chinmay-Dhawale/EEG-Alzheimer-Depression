
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mne
import joblib

from scipy.signal import welch, coherence

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

ALZ_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Alzheimer_vs_Healthy_Tuned_SVM.pkl"
)

DEP_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Depression_vs_Healthy_Tuned_SVM.pkl"
)

THREE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Three_Class_EEG_Tuned_SVM.pkl"
)


COMMON_CHANNELS = [
    "Fp1", "Fp2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "Fz", "Cz", "Pz"
]


BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}


# ============================================================
# CHANNEL NAME NORMALIZATION
# ============================================================

def normalize_channel_name(name):

    name = str(name).strip()

    # Depression dataset format:
    # EEG Fp1-LE -> Fp1
    if name.startswith("EEG "):
        name = name[4:]

    if "-" in name:
        name = name.split("-")[0]

    name = name.strip()

    replacements = {
        "T3": "T3",
        "T4": "T4",
        "T5": "T5",
        "T6": "T6"
    }

    return replacements.get(name, name)


# ============================================================
# LOAD EEG
# ============================================================

def load_eeg(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".edf":

        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            verbose=False
        )

    elif extension == ".set":

        raw = mne.io.read_raw_eeglab(
            file_path,
            preload=True,
            verbose=False
        )

    else:

        raise ValueError(
            "Unsupported file format. Please upload an EDF or SET file."
        )

    return raw


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_eeg(raw):

    original_sfreq = float(raw.info["sfreq"])
    original_channels = len(raw.ch_names)
    original_duration = raw.n_times / raw.info["sfreq"]

    # Normalize channel names
    rename_map = {}

    for ch in raw.ch_names:

        normalized = normalize_channel_name(ch)

        if normalized != ch:
            rename_map[ch] = normalized

    if rename_map:

        raw.rename_channels(rename_map)

    # Find required channels
    available = []

    for ch in COMMON_CHANNELS:

        if ch in raw.ch_names:
            available.append(ch)

    missing = [
        ch for ch in COMMON_CHANNELS
        if ch not in available
    ]

    if missing:

        raise ValueError(
            "The uploaded EEG does not contain all required channels.\n\n"
            "Missing channels:\n"
            + ", ".join(missing)
        )

    # Keep exact channel order
    raw.pick(COMMON_CHANNELS)

    # Resample to 256 Hz
    if abs(raw.info["sfreq"] - 256) > 0.01:

        raw.resample(
            256,
            npad="auto",
            verbose=False
        )

    # Bandpass filter
    raw.filter(
        l_freq=1.0,
        h_freq=45.0,
        verbose=False
    )

    # 50 Hz notch
    raw.notch_filter(
        freqs=[50],
        verbose=False
    )

    # Average reference
    raw.set_eeg_reference(
        "average",
        projection=False,
        verbose=False
    )

    return (
        raw,
        original_sfreq,
        original_channels,
        original_duration
    )


# ============================================================
# BAND POWER
# ============================================================

def calculate_band_power(signal, sfreq):

    nperseg = min(
        len(signal),
        int(sfreq * 4)
    )

    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=nperseg
    )

    total_mask = (
        (frequencies >= 1) &
        (frequencies <= 45)
    )

    total_power = np.trapezoid(
        psd[total_mask],
        frequencies[total_mask]
    )

    powers = {}

    for band, (low, high) in BANDS.items():

        mask = (
            (frequencies >= low) &
            (frequencies < high)
        )

        if np.any(mask):

            power = np.trapezoid(
                psd[mask],
                frequencies[mask]
            )

        else:

            power = 0.0

        powers[band] = power

    return powers, total_power


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(raw):

    sfreq = raw.info["sfreq"]

    data = raw.get_data()

    samples_per_window = int(
        sfreq * 10
    )

    total_samples = data.shape[1]

    n_windows = total_samples // samples_per_window

    clean_windows = []

    for i in range(n_windows):

        start = i * samples_per_window
        end = start + samples_per_window

        epoch = data[:, start:end]

        if epoch.shape[1] != samples_per_window:
            continue

        # 200 microvolt peak-to-peak rejection
        peak_to_peak = np.ptp(
            epoch,
            axis=1
        )

        if np.max(peak_to_peak) > 200e-6:
            continue

        clean_windows.append(epoch)

    if len(clean_windows) == 0:

        raise ValueError(
            "No clean 10-second EEG windows were found."
        )

    # --------------------------------------------------------
    # Feature containers
    # --------------------------------------------------------

    feature_values = []
    feature_names = []

    # --------------------------------------------------------
    # 1. Relative band power
    #    19 channels × 5 bands = 95
    # --------------------------------------------------------

    all_window_bandpowers = []

    for epoch in clean_windows:

        epoch_bandpowers = []

        for ch_idx, ch_name in enumerate(COMMON_CHANNELS):

            signal = epoch[ch_idx]

            powers, total_power = calculate_band_power(
                signal,
                sfreq
            )

            if total_power <= 0:
                relative = {
                    band: 0.0
                    for band in BANDS
                }

            else:
                relative = {
                    band: powers[band] / total_power
                    for band in BANDS
                }

            epoch_bandpowers.append(relative)

        all_window_bandpowers.append(
            epoch_bandpowers
        )

    all_window_bandpowers = np.array(
        [
            [
                [
                    item[band]
                    for band in BANDS
                ]
                for item in window
            ]
            for window in all_window_bandpowers
        ]
    )

    # Shape:
    # windows × channels × bands

    mean_bandpowers = np.mean(
        all_window_bandpowers,
        axis=0
    )

    for ch_idx, ch_name in enumerate(COMMON_CHANNELS):

        for band_idx, band in enumerate(BANDS):

            feature_names.append(
                f"{ch_name}_{band}_RelativePower"
            )

            feature_values.append(
                mean_bandpowers[ch_idx, band_idx]
            )

    # --------------------------------------------------------
    # 2. Global band powers = 5
    # --------------------------------------------------------

    global_bandpowers = np.mean(
        mean_bandpowers,
        axis=0
    )

    for band_idx, band in enumerate(BANDS):

        feature_names.append(
            f"Global_{band}_RelativePower"
        )

        feature_values.append(
            global_bandpowers[band_idx]
        )

    # --------------------------------------------------------
    # 3. Alpha hemispheric asymmetry = 8
    # --------------------------------------------------------

    asymmetry_pairs = [
        ("Fp1", "Fp2"),
        ("F3", "F4"),
        ("F7", "F8"),
        ("C3", "C4"),
        ("P3", "P4"),
        ("O1", "O2"),
        ("T3", "T4"),
        ("T5", "T6")
    ]

    channel_index = {
        ch: idx
        for idx, ch in enumerate(COMMON_CHANNELS)
    }

    alpha_idx = list(BANDS.keys()).index("Alpha")

    for left, right in asymmetry_pairs:

        left_power = mean_bandpowers[
            channel_index[left],
            alpha_idx
        ]

        right_power = mean_bandpowers[
            channel_index[right],
            alpha_idx
        ]

        asymmetry = np.log(
            (left_power + 1e-12) /
            (right_power + 1e-12)
        )

        feature_names.append(
            f"Alpha_Asymmetry_{left}_{right}"
        )

        feature_values.append(
            asymmetry
        )

    # --------------------------------------------------------
    # 4. O1-Fz and O2-Fz alpha coherence = 2
    # --------------------------------------------------------

    o1_idx = channel_index["O1"]
    o2_idx = channel_index["O2"]
    fz_idx = channel_index["Fz"]

    o1_coherences = []
    o2_coherences = []

    for epoch in clean_windows:

        o1 = epoch[o1_idx]
        o2 = epoch[o2_idx]
        fz = epoch[fz_idx]

        f_coh, coh_o1 = coherence(
            o1,
            fz,
            fs=sfreq,
            nperseg=min(
                len(o1),
                int(sfreq * 4)
            )
        )

        _, coh_o2 = coherence(
            o2,
            fz,
            fs=sfreq,
            nperseg=min(
                len(o2),
                int(sfreq * 4)
            )
        )

        alpha_mask = (
            (f_coh >= 8) &
            (f_coh <= 13)
        )

        if np.any(alpha_mask):

            o1_coherences.append(
                np.mean(coh_o1[alpha_mask])
            )

            o2_coherences.append(
                np.mean(coh_o2[alpha_mask])
            )

    feature_names.append(
        "O1_Fz_Alpha_Coherence"
    )

    feature_values.append(
        np.mean(o1_coherences)
    )

    feature_names.append(
        "O2_Fz_Alpha_Coherence"
    )

    feature_values.append(
        np.mean(o2_coherences)
    )

    # --------------------------------------------------------
    # Create final feature dataframe
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        [feature_values],
        columns=feature_names
    )

    return (
        feature_df,
        n_windows,
        len(clean_windows),
        all_window_bandpowers
    )


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    alz_model = joblib.load(
        ALZ_MODEL_PATH
    )

    dep_model = joblib.load(
        DEP_MODEL_PATH
    )

    three_model = joblib.load(
        THREE_MODEL_PATH
    )

    return (
        alz_model,
        dep_model,
        three_model
    )


# ============================================================
# PREDICTION
# ============================================================

def get_prediction(model, features):

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    classes = list(
        model.classes_
    )

    probability_dict = {
        str(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities
        )
    }

    return (
        str(prediction),
        probability_dict
    )


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="EEG Alzheimer's & Depression AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "EEG Alzheimer's & Depression AI"
)

st.subheader(
    "EEG Signal Machine Learning Classification"
)

st.write(
    "Upload an EEG recording in EDF or SET format. "
    "The system preprocesses the EEG, extracts "
    "frequency-domain features, and applies trained "
    "machine-learning models."
)

st.info(
    "This is an academic EEG machine-learning project. "
    "The outputs are model classification results and "
    "should not be interpreted as a medical diagnosis."
)


# ============================================================
# MODEL CHECK
# ============================================================

try:

    alz_model, dep_model, three_model = load_models()

except Exception as e:

    st.error(
        "Unable to load the trained models."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload EEG Recording",
    type=["edf", "set"]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.write(
        f"**Selected file:** {uploaded_file.name}"
    )

    if st.button(
        "ANALYZE EEG",
        type="primary"
    ):

        temp_path = None

        try:

            # ------------------------------------------------
            # Save uploaded file temporarily
            # ------------------------------------------------

            extension = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as tmp:

                tmp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = tmp.name

            # ------------------------------------------------
            # Load
            # ------------------------------------------------

            with st.spinner(
                "Loading EEG recording..."
            ):

                raw = load_eeg(
                    temp_path
                )

            # ------------------------------------------------
            # Original information
            # ------------------------------------------------

            original_sfreq = float(
                raw.info["sfreq"]
            )

            original_channels = len(
                raw.ch_names
            )

            original_duration = (
                raw.n_times /
                raw.info["sfreq"]
            )

            # ------------------------------------------------
            # Preprocessing
            # ------------------------------------------------

            with st.spinner(
                "Preprocessing EEG..."
            ):

                (
                    processed_raw,
                    original_sfreq,
                    original_channels,
                    original_duration
                ) = preprocess_eeg(
                    raw
                )

            # ------------------------------------------------
            # Feature extraction
            # ------------------------------------------------

            with st.spinner(
                "Extracting EEG features..."
            ):

                (
                    features,
                    total_windows,
                    clean_windows,
                    window_bandpowers
                ) = extract_features(
                    processed_raw
                )

            # ------------------------------------------------
            # Ensure feature count
            # ------------------------------------------------

            if features.shape[1] != 110:

                raise ValueError(
                    f"Expected 110 EEG features, "
                    f"but extracted {features.shape[1]}."
                )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            with st.spinner(
                "Running machine-learning models..."
            ):

                alz_prediction, alz_prob = get_prediction(
                    alz_model,
                    features
                )

                dep_prediction, dep_prob = get_prediction(
                    dep_model,
                    features
                )

                three_prediction, three_prob = get_prediction(
                    three_model,
                    features
                )

            # =================================================
            # EEG INFORMATION
            # =================================================

            st.divider()

            st.header(
                "EEG Analysis Report"
            )

            st.subheader(
                "EEG Information"
            )

            info_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Original Sampling Rate",
                        "Original Channels",
                        "Original Duration",
                        "Processed Sampling Rate",
                        "Processed Channels",
                        "Window Size",
                        "Total Windows",
                        "Clean Windows",
                        "Number of Features"
                    ],
                    "Value": [
                        f"{original_sfreq:.1f} Hz",
                        original_channels,
                        f"{original_duration:.1f} seconds",
                        f"{processed_raw.info['sfreq']:.0f} Hz",
                        len(processed_raw.ch_names),
                        "10 seconds",
                        total_windows,
                        clean_windows,
                        features.shape[1]
                    ]
                }
            )

            st.table(
                info_df
            )

            # =================================================
            # ALZHEIMER
            # =================================================

            st.divider()

            st.subheader(
                "Alzheimer's vs Healthy"
            )

            if alz_prediction.lower() in [
                "alzheimer",
                "alzheimer's",
                "a"
            ]:

                st.warning(
                    f"Model prediction: {alz_prediction}"
                )

            else:

                st.success(
                    f"Model prediction: {alz_prediction}"
                )

            alz_prob_df = pd.DataFrame(
                {
                    "Class": list(
                        alz_prob.keys()
                    ),
                    "Probability": [
                        f"{v * 100:.2f}%"
                        for v in alz_prob.values()
                    ]
                }
            )

            st.table(
                alz_prob_df
            )

            # =================================================
            # DEPRESSION
            # =================================================

            st.subheader(
                "Depression vs Healthy"
            )

            if dep_prediction.lower() in [
                "depression",
                "mdd",
                "m"
            ]:

                st.warning(
                    f"Model prediction: {dep_prediction}"
                )

            else:

                st.success(
                    f"Model prediction: {dep_prediction}"
                )

            dep_prob_df = pd.DataFrame(
                {
                    "Class": list(
                        dep_prob.keys()
                    ),
                    "Probability": [
                        f"{v * 100:.2f}%"
                        for v in dep_prob.values()
                    ]
                }
            )

            st.table(
                dep_prob_df
            )

            # =================================================
            # THREE CLASS
            # =================================================

            st.subheader(
                "Three-Class Model"
            )

            st.write(
                f"**Prediction:** {three_prediction}"
            )

            three_prob_df = pd.DataFrame(
                {
                    "Class": list(
                        three_prob.keys()
                    ),
                    "Probability": [
                        f"{v * 100:.2f}%"
                        for v in three_prob.values()
                    ]
                }
            )

            st.table(
                three_prob_df
            )

            # =================================================
            # EEG SIGNAL PLOT
            # =================================================

            st.divider()

            st.subheader(
                "Preprocessed EEG Signal — First 10 Seconds"
            )

            plot_data = processed_raw.get_data()

            samples_to_plot = min(
                plot_data.shape[1],
                int(
                    processed_raw.info["sfreq"] * 10
                )
            )

            time_axis = np.arange(
                samples_to_plot
            ) / processed_raw.info["sfreq"]

            fig, ax = plt.subplots(
                figsize=(14, 8)
            )

            # Plot first 8 channels
            channels_to_plot = min(
                8,
                len(COMMON_CHANNELS)
            )

            for idx in range(
                channels_to_plot
            ):

                signal = (
                    plot_data[
                        idx,
                        :samples_to_plot
                    ]
                    * 1e6
                )

                offset = idx * 100

                ax.plot(
                    time_axis,
                    signal + offset,
                    linewidth=0.7,
                    label=COMMON_CHANNELS[idx]
                )

            ax.set_xlabel(
                "Time (seconds)"
            )

            ax.set_ylabel(
                "Amplitude (µV) + offset"
            )

            ax.set_title(
                "Preprocessed EEG — First 10 Seconds"
            )

            ax.legend(
                loc="upper right",
                fontsize=8
            )

            ax.grid(
                alpha=0.25
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            # =================================================
            # FREQUENCY BAND POWER
            # =================================================

            st.subheader(
                "EEG Frequency Band Features"
            )

            average_band_power = np.mean(
                window_bandpowers,
                axis=(0, 1)
            )

            band_names = list(
                BANDS.keys()
            )

            band_power_df = pd.DataFrame(
                {
                    "Frequency Band": band_names,
                    "Relative Power (%)": (
                        average_band_power * 100
                    )
                }
            )

            st.bar_chart(
                band_power_df.set_index(
                    "Frequency Band"
                )
            )

            st.dataframe(
                band_power_df,
                use_container_width=True
            )

            # =================================================
            # FEATURE TABLE
            # =================================================

            st.subheader(
                "Extracted EEG Features"
            )

            feature_display = features.T.reset_index()

            feature_display.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                feature_display,
                use_container_width=True
            )

            # =================================================
            # DOWNLOAD REPORT
            # =================================================

            report = features.copy()

            report["Alzheimer_Prediction"] = (
                alz_prediction
            )

            report["Depression_Prediction"] = (
                dep_prediction
            )

            report["Three_Class_Prediction"] = (
                three_prediction
            )

            csv_data = report.to_csv(
                index=False
            )

            st.download_button(
                label="Download EEG Analysis CSV",
                data=csv_data,
                file_name="EEG_analysis_report.csv",
                mime="text/csv"
            )

            # =================================================
            # PROJECT PIPELINE
            # =================================================

            st.divider()

            st.subheader(
                "Project Pipeline"
            )

            st.write(
                "EEG Upload → Preprocessing → "
                "10-second Segmentation → "
                "Feature Extraction → "
                "SVM Models → Classification Report"
            )

            st.success(
                "EEG analysis completed successfully."
            )

        except Exception as e:

            st.error(
                "EEG analysis failed."
            )

            st.exception(e)

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                try:
                    os.remove(temp_path)
                except:
                    pass
