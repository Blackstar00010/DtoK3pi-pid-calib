import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from torch import nn, optim
import torch
from src.utils import utils, dataprep, config


def filter_outliers(df: pd.DataFrame, gmm_percent=1, iqr_factor=4) -> pd.DataFrame:
    """
    Filter out the worst 1% of the data using Mahalanobis distance
    Args:
        df: DataFrame. The data to filter.
        gmm_percent: int. The percentage of data to keep.
        iqr_factor: int. The factor to multiply the IQR by to determine the range to keep. If 0, no IQR filtering.

    Returns:
        pd.DataFrame. The filtered data.
    """
    # temp = df.copy()  # for debugging
    n_components = 1
    gmm = GaussianMixture(n_components=n_components, reg_covar=1e-3)
    gmm.fit(df)
    center = gmm.means_[0]
    if gmm.covariance_type == 'full':
        # Full covariance: K x D x D
        stddevs = np.sqrt(np.array([np.diag(cov) for cov in gmm.covariances_]))
    elif gmm.covariance_type == 'diag':
        # Diagonal covariance: K x D
        stddevs = np.sqrt(gmm.covariances_)
    elif gmm.covariance_type == 'tied':
        # Tied covariance: D x D
        stddevs = np.sqrt(np.diag(gmm.covariances_))
    elif gmm.covariance_type == 'spherical':
        # Spherical covariance: K
        stddevs = np.sqrt(gmm.covariances_[:, np.newaxis])
    else:
        raise ValueError(f"Unknown covariance type: {gmm.covariance_type}")
    # d^2 = sum_i [ (x_i - center_i)^2 / stddev_i^2 ]
    mahalanobis = np.sum(((df - center) / stddevs) ** 2, axis=1)
    threshold = np.percentile(mahalanobis, 100 - gmm_percent)
    df = df[mahalanobis < threshold]

    # IQR
    if iqr_factor != 0:
        iqr = df.quantile(0.75) - df.quantile(0.25)
        med = df.median()
        truth_df = (df > med - iqr_factor * iqr) & (df < med + iqr_factor * iqr)
        very_bad_iqr = iqr == 0
        truth_df.loc[:, very_bad_iqr] = True
        bad_iqr = truth_df.columns[truth_df.sum() / len(truth_df) < 0.9]
        truth_df.loc[:, bad_iqr] = True
        raise NotImplementedError("IQR filtering is not implemented yet.")
        df = df[truth_df.all(axis=1)]
        if len(df) < len(temp) * 0.9:  # copilot says 0.9 but the actual data is something like 0.03 lol
            raise ValueError("Too many outliers removed!")
    return df


def preprocess(df: pd.DataFrame, log_scale=True) -> tuple:
    """
    Preprocess the data for GAN training. Preprocessing includes:\n
    - Drop columns with more than 50% NaN values
    - Drop columns with only one unique value
    - Apply log transformation to columns with high skewness
    - Normalize the data to [-1, 1]
    Args:
        df: pd.DataFrame. The data to preprocess.
        log_scale: bool. Whether to apply log transformation to columns with high skewness.

    Returns:
        tuple. The preprocessed data, the original minimum and maximum values,
        the original data types, the boolean columns, the high skew columns, and the low skew columns.
    """
    og_cols = df.columns

    # Drop columns with more than 50% NaN values
    # df = df.replace([np.inf, -np.inf], np.nan)
    # df = df.dropna(thresh=int(0.5 * len(df)), axis=1)
    # bad_cols = df.columns[df.isna().any()]
    # full_cols = utils.load('full.root').columns
    # very_bad_cols = [col for col in bad_cols if col not in full_cols]
    # df = df.drop(very_bad_cols, axis=1)
    # df = df.dropna()
    # any -> 44475, 886; 50% -> 43724, 889; 90% -> 43724, 889

    # Drop columns with only one unique value i.e. pointless
    obvious_cols = df.columns[df.nunique() == 1]
    df = df.drop(obvious_cols, axis=1)
    # also obvious
    df = df.drop(columns=[col for col in df.columns if col.endswith('PARTICLE_ID')])

    # Types
    obj_cols = df.columns[df.dtypes == 'object']
    df = df.drop(obj_cols, axis=1)
    og_types = df.dtypes
    bool_cols = df.columns[df.dtypes == 'bool']
    df_bool = df[bool_cols].astype(int)
    df_num = df.drop(bool_cols, axis=1)

    # id_cols = [item for item in df_num.columns if item.endswith('TRUEID') or item.endswith("MOTHER_ID")]
    # df_id = df_num[id_cols]
    # df_num = df_num.drop(id_cols, axis=1)
    # signs = np.sign(df_id)
    # df_id = df_id.abs()
    # uniques = df_id.stack().unique()
    # id_mapping = {item: i for i, item in enumerate(uniques)}
    # df_id = df_id.replace(id_mapping)
    # df_id = df_id * signs
    id_cols = pd.Index([])
    df_id = pd.DataFrame()
    id_mapping = None  # already filtered by dataprep.filter_mc function

    discrete_cols = [item for item in df_num.columns if df_num[item].abs().nunique() < 100]
    df_disc = df_num[discrete_cols]
    df_num = df_num.drop(discrete_cols, axis=1)
    signs = np.sign(df_disc)
    df_disc = df_disc.abs()
    uniques = df_disc.stack().unique()
    disc_mapping = {item: i for i, item in enumerate(uniques)}
    df_disc = df_disc.replace(disc_mapping)
    df_disc = df_disc * signs

    # TODO: DO SOME MORE ENCODING BY ANALYSING & DEBUGGING filter_outliers FUNCTION

    # Filter out the farthest 10% of the data using Mahalanobis distance
    temp1 = df_num.copy()
    m = 'Dst_MINIPCHI2'
    df_num = filter_outliers(df_num, 10, 0)

    # Drop columns with only one unique value, again in case outliers removed non-unique values
    obvious_cols = df_num.columns[df_num.nunique() == 1]
    df_num = df_num.drop(obvious_cols, axis=1)
    obvious_cols = df_bool.columns[df_bool.nunique() == 1]
    df_bool = df_bool.drop(obvious_cols, axis=1)

    temp2 = df_num.copy()
    if log_scale:
        # Apply log transformation to columns with high skewness
        cut = 10
        skewness = df_num.skew()
        high_skew = skewness[skewness > cut].index  # = names of columns that are highly skewed to the right /\_
        offset = df_num[high_skew].min()
        df_num[high_skew] = np.log1p(df_num[high_skew] - offset)  # log1p(x) = log(1 + x)
        low_skew = skewness[skewness < -cut].index  # = names of columns that are highly skewed to the left _/\
        # exclude those with max() > 10 as they produce inf values
        low_skew = low_skew[df_num[low_skew].max() < 10]
        df_num[low_skew] = np.expm1(df_num[low_skew])  # expm1(x) = exp(x) - 1
    else:
        high_skew = low_skew = pd.Index([])
        offset = 0

    # Normalize the data to [-1, 1]
    og_min, og_max = df_num.min(), df_num.max()
    df_num = (df_num - og_min) / (og_max - og_min) * 2 - 1  # Normalize to [-1, 1]

    # merge back the bool columns
    df = pd.concat([df_num, df_bool, df_disc, df_id], axis=1)
    df = df.dropna()
    df = df.drop(columns=df.columns[df.nunique() == 1])  # drop columns with only one unique value

    # columns order
    df = df[[col for col in og_cols if col in df.columns]]

    # prepare returning
    og_stats = {'min': og_min, 'max': og_max, 'types': og_types, 'cols': df_num.columns}
    spec_cols = {'bool': bool_cols, "id_cols": id_cols, 'discrete': discrete_cols}
    maps = {'id': id_mapping, 'disc': disc_mapping}
    skews = {'high': high_skew, 'offset': offset, 'low': low_skew}
    return df, og_stats, spec_cols, maps, skews

# Define Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        # latent_dim ~ 16, data_dim ~ 889
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh()  # Outputs in the range [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Outputs a probability
        )

    def forward(self, x):
        return self.model(x)


class MyGAN:
    def __init__(self, epochs=300):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        utils.log(f"Using device: {self.device}")
        self.generator = self.discriminator = None

        # hyperparameters
        self.latent_dim = 16
        self.data_dim = None
        self.batch_size = 4096  # approx. 1/10 of the data
        self.epochs = epochs
        self.lr = 0.0002

        # information for original df
        self.columns = self.og_stats = self.spec_cols = self.maps = self.skews = None

    def fit(self, original_df: pd.DataFrame):
        original_df = dataprep.filter_mc(original_df,
                                         apply_trueid=True, apply_mothertrueid=True, apply_bkgcat=False,
                                         cut_cols=False)
        id_cols = [col for col in original_df.columns if col.endswith('TRUEID') or col.endswith("MOTHER_ID")]
        original_df = original_df.drop(columns=id_cols)
        # temp = df.copy()  # for debugging
        og_shape = original_df.shape
        original_df, self.og_stats, self.spec_cols, self.maps, self.skews = preprocess(original_df)
        utils.log(f"Data preprocessed! Shape: {og_shape} -> {original_df.shape}")
        data = original_df.values
        # preprocess(temp)  # for debugging

        # Convert data to PyTorch tensor
        self.columns = original_df.columns
        self.data_dim = len(self.columns)
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
        data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=self.batch_size, shuffle=True)

        # Initialize models
        self.generator = Generator(self.latent_dim, self.data_dim).to(self.device)
        self.discriminator = Discriminator(self.data_dim).to(self.device)

        # Loss and optimizers
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            d_loss = g_loss = 0
            for real_data in data_loader:
                # Train Discriminator
                optimizer_d.zero_grad()
                real_data = real_data.to(self.device)
                real_labels = torch.ones(real_data.size(0), 1).to(self.device)
                fake_labels = torch.zeros(real_data.size(0), 1).to(self.device)

                # Real data
                real_output = self.discriminator(real_data)
                real_loss = criterion(real_output, real_labels)

                # Fake data
                z = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
                fake_data = self.generator(z).detach()  # Detach to avoid training generator here
                fake_output = self.discriminator(fake_data)
                fake_loss = criterion(fake_output, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                z = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
                generated_data = self.generator(z)
                fake_output = self.discriminator(generated_data)
                g_loss = criterion(fake_output, real_labels)  # Fool discriminator

                g_loss.backward()
                optimizer_g.step()

            # log losses
            if epoch % 3 == 0 or epoch == self.epochs - 1:
                utils.log(f"Epoch [{epoch}/{self.epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    def sample(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        synthetic_data = self.generator(z).detach().cpu().numpy()

        # og_stats = {'min': og_min, 'max': og_max, 'types': og_types, 'cols': df_num.columns}
        # spec_cols = {'bool': bool_cols, "id_cols": id_cols, 'discrete': discrete_cols}
        # maps = {'id': id_mapping, 'disc': disc_mapping}
        # skews = {'high': high_skew, 'offset': offset, 'low': low_skew}

        # Rescale synthetic data back to the original scale (from [-1, 1] to [df.min(), df.max()])
        # synthetic_df = pd.DataFrame(synthetic_data, columns=original_df.columns)
        synthetic_df = pd.DataFrame(synthetic_data, columns=self.columns)
        synthetic_df = synthetic_df * 0.5 + 0.5  # Rescale to [0, 1]
        og_max, og_min = self.og_stats['max'], self.og_stats['min']
        synthetic_df[self.og_stats['cols']] = synthetic_df[self.og_stats['cols']] * (og_max - og_min) + og_min

        # apply high_skew and low_skew
        high_scew, offset, low_skew = self.skews['high'], self.skews['offset'], self.skews['low']
        synthetic_df[high_scew] = np.expm1(synthetic_df[high_scew]) + offset
        synthetic_df[low_skew] = np.log1p(synthetic_df[low_skew])

        # apply id_mapping and disc_mapping
        id_mapping, disc_mapping = self.maps['id'], self.maps['disc']
        if id_mapping is not None:
            raise NotImplementedError("ID mapping is not implemented yet.")
        df_disc = synthetic_df[self.spec_cols['discrete']]
        disc_signs = np.sign(df_disc)
        df_disc = df_disc.abs()

        def find_closest(x, mapping: dict):
            return min(mapping, key=lambda y: abs(x - y))

        df_disc = df_disc.map(lambda x: find_closest(x, disc_mapping))
        df_disc = df_disc.replace({v: k for k, v in disc_mapping.items()})
        df_disc = df_disc * disc_signs
        synthetic_df[self.spec_cols['discrete']] = df_disc

        # apply og_types
        bool_cols, og_types = self.spec_cols['bool'], self.og_stats['types']
        alive_bool_cols = bool_cols[bool_cols.isin(synthetic_df.columns)]
        synthetic_df[alive_bool_cols] = synthetic_df[alive_bool_cols].round().astype(bool)
        alive_og_types = og_types[og_types.index.isin(synthetic_df.columns)]
        synthetic_df = synthetic_df.astype(alive_og_types)
        # column sorting
        synthetic_df = synthetic_df[self.columns]
        return synthetic_df

    def save(self, common_name=f"{config.generators_dir}/gan"):
        torch.save(self.generator.state_dict(), f"{common_name}_g.pth")
        torch.save(self.discriminator.state_dict(), f"{common_name}_d.pth")
        utils.log("Models saved!")

    def load(self, common_name=f"{config.generators_dir}/gan"):
        self.generator.load_state_dict(torch.load(f"{common_name}_g.pth"))
        self.discriminator.load_state_dict(torch.load(f"{common_name}_d.pth"))
        utils.log("Models loaded!")