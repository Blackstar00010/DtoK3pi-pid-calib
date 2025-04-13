import pandas as pd
import numpy as np
from src.utils import utils
from src.utils import plotter, config
from myGenerators import MyGAN
from ctgan import CTGAN
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from realtabformer import REaLTabFormer
from scipy.stats import ks_2samp


def generate_gan(original_df: pd.DataFrame, n_samples=1000, epochs=300) -> None:
    """
    Generates synthetic data using a GAN model
    Args:
        original_df: The original data
        n_samples: The number of samples(rows) to generate
        epochs: The number of epochs to train the model

    Returns:
        None
    """
    utils.log("Initialising GAN...")
    model = MyGAN(epochs)
    utils.log("Fitting the model...")
    model.fit(original_df)
    utils.log("Sampling the data...")
    synthetic_data = model.sample(n_samples)
    utils.log("Saving the synthetic data and model...")
    synthetic_data.to_csv(f"{config.synmc_dir}/gan.csv", index=False)
    model.save(f"{config.generators_dir}/gan")
    utils.log("Finished generating synthetic data using GAN.")


def generate_ctgan(original_df: pd.DataFrame, n_samples=1000, epochs=300) -> None:
    """
    Generates synthetic data using CTGAN model (https://github.com/sdv-dev/CTGAN)
    Args:
        original_df: The original data
        n_samples: The number of samples(rows) to generate
        epochs: The number of epochs to train the model

    Returns:
        None
    """
    utils.log("Initialising CTGAN...")
    model = CTGAN(epochs=epochs, verbose=True, cuda=True)
    utils.log("Fitting the model...")
    model.fit(original_df)
    utils.log("Sampling the data...")
    synthetic_data = model.sample(n_samples)
    utils.log("Saving the synthetic data and model...")
    synthetic_data.to_csv(f"{config.synmc_dir}/ctgan.csv", index=False)
    model.save(f"{config.generators_dir}/ctgan.pkl")
    utils.log("Finished generating synthetic data using CTGAN.")


def generate_sdv_models(original_df: pd.DataFrame, n_samples=1000) -> None:
    """
    Generates synthetic data using models made by Synthetic Data Vault (https://docs.sdv.dev/sdv/)
    Args:
        original_df: The original data
        n_samples: The number of samples(rows) to generate

    Returns:
        None
    """
    metadata = Metadata.detect_from_dataframe(original_df)
    for model_class, name in [(CTGANSynthesizer, "ctgan"), (GaussianCopulaSynthesizer, "copula"), (TVAESynthesizer, "tvae")]:
        utils.log(f"Initialising {name}...")
        model = model_class(metadata, verbose=True)
        utils.log(f"Fitting the {name} model...")
        model.fit(original_df)
        utils.log(f"Sampling the data...")
        synthetic_data = model.sample(n_samples)
        utils.log(f"Saving the synthetic data and model...")
        synthetic_data.to_csv(f"{config.synmc_dir}/{name}_sdv.csv", index=False)
        model.save(f"{config.generators_dir}/{name}_sdv.pkl")
        utils.log(f"Finished generating synthetic data using {name}.")

    # to load models, use `model = XXXX.load('path/to/model.pkl')`


def generate_tvae():
    pass


def generate_rtf(original_df: pd.DataFrame, n_samples=1000) -> None:
    """
    Generates synthetic data using REaLTabFormer by WorldBank(https://github.com/worldbank/REaLTabFormer) which uses transformers
    Args:
        original_df: The original data
        n_samples: The number of samples(rows) to generate

    Returns:
        None
    """
    utils.log("Initialising REaLTabFormer...")
    rtf_model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=2,
        batch_size=4,
        logging_steps=3,
        train_size=0.8,)
    utils.log("Fitting the model...")
    rtf_model.fit(original_df)
    utils.log("Sampling the data...")
    samples = rtf_model.sample(n_samples=n_samples)
    utils.log("Saving the synthetic data and model...")
    samples.to_csv(f"{config.synmc_dir}/rtf.csv", index=False)
    rtf_model.save(config.generators_dir)
    utils.log("Finished generating synthetic data using REaLTabFormer.")

    # In it, a directory with the model's experiment id `idXXXX` will also be created where the artefacts of the model will be stored.
    # To load the model, use this line of code:
    # rtf_model2 = REaLTabFormer.load_from_dir(path="rtf_model/idXXXX")


def plot_comparison(original_df: pd.DataFrame=None) -> None:
    """
    Plots the distributions of the original and synthetic data and saves the plots at plot_dirs[0] specified in config.py
    Args:
        original_df: The original data

    Returns:
        None
    """
    og_df = pd.read_csv(f"{config.mc_dir}/mc.csv") if original_df is None else original_df
    for syn_file in utils.listdir(config.synmc_dir, end=".csv"):
        utils.log(f"Plotting quality comparison for {syn_file}...")
        syn_df = pd.read_csv(f"{config.synmc_dir}/{syn_file}")

        # Plot the distributions of the original and synthetic data
        for col in syn_df.columns:
            to_plot = pd.DataFrame()
            to_plot['Original'] = og_df[col]
            to_plot['Synthetic'] = syn_df[col]
            ks_result = ks_2samp(og_df[col], syn_df[col])
            # to_plot = pd.DataFrame({"Original": og_df[col], "Synthetic": syn_df[col]})
            plotter.plot_1dhist(to_plot, bins=100, log=False, density=True, style="alpha", x_label=col,
                                title=f"KS: {ks_result.statistic:.2f}, p-value: {ks_result.pvalue:.2f}",
                                filename=f"{config.plot_dirs[0]}/{syn_file.replace('.csv', '')}/{col}.png")
            utils.log(f"Finished plotting quality comparison for {col}.")
        utils.log(f"Finished plotting quality comparison for {syn_file}.")

        # sig_test = np.pad(sig_test, (0, longer_len - len(sig_test)), constant_values=np.nan)
        # bkg_test = np.pad(bkg_test, (0, longer_len - len(bkg_test)), constant_values=np.nan)
        # to_plot = pd.DataFrame({"sig": sig_test, "bkg": bkg_test})
        # plot_1dhist(to_plot, bins=100, log=False, density=True, style="alpha",
        #             x_label="Probability", filename=filename)


def preprocess_mc(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(thresh=int(0.5 * len(df)), axis=1)
    bad_cols = df.columns[df.isna().any()]
    full_cols = utils.load('full.root').columns
    very_bad_cols = [col for col in bad_cols if col not in full_cols]
    df = df.drop(very_bad_cols, axis=1)
    df = df.dropna()
    return df


@utils.alert
def main():
    utils.log(f'Running {__file__}...')
    mc_df = pd.read_csv(f"{config.mc_dir}/mc.csv")
    mc_df = preprocess_mc(mc_df)
    # generate_gan(mc_df)
    # generate_rtf(mc_df)
    generate_ctgan(mc_df)
    # generate_sdv_models(mc_df)
    plot_comparison(mc_df)


if __name__ == '__main__':
    main()
