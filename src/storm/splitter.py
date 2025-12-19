from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import numpy as np
import random
from omegaconf import DictConfig
import os

## toml splitter
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # fallback, if needed
    import tomli as tomllib


# module-level logger
import logging
log = logging.getLogger(__name__)

# copied from perturbench
class PerturbationDataSplitter:
    """Class to split data into train/test/test.

    Attributes:
        obs_dataframe: Dataframe with cell/sample level metadata.
          Must contain a column with the perturbation key
        perturbation_key: Column name of containing the perturbation identity
          of each cell/sample
        covariate_keys:  Column name(s) of the covariates to split on
          (i.e. `cell_type`). If None, a dummy covariate will be created
        perturbation_control_value: Identity of control perturbation
        perturbation_combination_delimiter: Delimiter separating multiple
          perturbations.
    """

    obs_dataframe: pd.DataFrame
    perturbation_key: str
    perturbation_control_value: str
    covariate_keys: list[str] | str | None = None
    perturbation_combination_delimiter: str = "+"

    @staticmethod
    def split_dataset(
        splitter_config: DictConfig,
        obs_dataframe: pd.DataFrame,
        perturbation_key: str,
        perturbation_combination_delimiter: str,
        perturbation_control_value: str,
    ):
        """Split dataset into train/val/test depending on task specified in
           splitter config.

        Args
            splitter_config: Configuration for the splitter
            obs_dataframe: Dataframe with cell/sample level metadata.
            perturbation_key: Dataframe column containing the perturbation
            perturbation_combination_delimiter: Delimiter separating multiple
                perturbations.
            perturbation_control_value: Identity of control perturbation

        Returns:
            split_dict: Dictionary with keys 'train', 'val', 'test' and values
                as numpy arrays of indices for each split
        """
        # AnnData Split
        if "split_path" in splitter_config:
            split = pd.read_csv(
                splitter_config.split_path, index_col=0, header=None
            ).iloc[:, 0]
        else:
            perturbation_datasplitter = PerturbationDataSplitter(
                obs_dataframe=obs_dataframe,
                perturbation_key=perturbation_key,
                covariate_keys=list(splitter_config.covariate_keys),
                perturbation_control_value=perturbation_control_value,
                perturbation_combination_delimiter=perturbation_combination_delimiter,
            )
            if splitter_config.task == "transfer":
                split = perturbation_datasplitter.split_covariates(
                    seed=splitter_config.splitter_seed,
                    min_train_covariates=splitter_config.min_train_covariates,
                    max_heldout_covariates=splitter_config.max_heldout_covariates,
                    max_heldout_fraction_per_covariate=splitter_config.max_heldout_fraction_per_covariate,
                    train_control_fraction=splitter_config.train_control_fraction,
                    downsample_fraction=splitter_config.downsample_fraction,
                )
            elif splitter_config.task == "combine":
                split = perturbation_datasplitter.split_combinations(
                    seed=splitter_config.splitter_seed,
                    max_heldout_fraction_per_covariate=splitter_config.max_heldout_fraction_per_covariate,
                    train_control_fraction=splitter_config.train_control_fraction,
                    downsample_fraction=splitter_config.downsample_fraction,
                )
            elif splitter_config.task == "combine_inverse":
                split = perturbation_datasplitter.split_combinations_inverse(
                    seed=splitter_config.splitter_seed,
                    max_heldout_fraction_per_covariate=splitter_config.max_heldout_fraction_per_covariate,
                    train_control_fraction=splitter_config.train_control_fraction,
                    downsample_fraction=splitter_config.downsample_fraction,
                )
            else:
                raise ValueError(
                    'splitter_config.task must be "transfer", "combine", or "combine_inverse"'
                )

        assert len(split) == obs_dataframe.shape[0]
        assert split.index.equals(obs_dataframe.index)
        for split_value in ["train", "val", "test"]:
            assert split_value in split.unique()

        if splitter_config.get("save"):
            if not os.path.exists(splitter_config.output_path):
                os.makedirs(splitter_config.output_path)
            try:
                split.to_csv(
                    splitter_config.output_path + "train_test_split.csv", index=True
                )
            except PermissionError:
                print(f"Warning: Unable to save split to {splitter_config.output_path}")

        split_dict = {
            split_val: np.where(split == split_val)[0]
            for split_val in ["train", "val", "test"]
        }
        return split_dict

    def __init__(
        self,
        obs_dataframe: pd.DataFrame,
        perturbation_key: str,
        perturbation_control_value: str,
        covariate_keys: list[str] | str | None = None,
        perturbation_combination_delimiter: str = "+",
    ):
        """Initialize PerturbationDataSplitter object."""
        self.obs_dataframe = obs_dataframe
        self.perturbation_key = perturbation_key

        assert (
            perturbation_control_value in self.obs_dataframe[perturbation_key].unique()
        )

        if covariate_keys is None:
            obs_dataframe["dummy_cov"] = "1"
            self.covariate_keys = ["dummy_cov"]
        elif isinstance(covariate_keys, str):
            self.covariate_keys = [covariate_keys]
        else:
            self.covariate_keys = covariate_keys

        self.perturbation_control_value = perturbation_control_value
        self.perturbation_combination_delimiter = perturbation_combination_delimiter

        self.split_params = {}
        self.summary_dataframes = {}

        covariates_list = [
            list(obs_dataframe[covariate_key].values)
            for covariate_key in self.covariate_keys
        ]
        self.covariates_merged = [
            frozenset(covariates) for covariates in zip(*covariates_list)
        ]
        self.perturbation_covariates = [
            (perturbation, covariates)
            for perturbation, covariates in zip(
                obs_dataframe[perturbation_key], self.covariates_merged
            )
        ]

    def _assign_split(
        self,
        seed: int,
        train_perturbation_covariates: list[tuple[str, frozenset[str]]],
        heldout_perturbation_covariates: list[tuple[str, frozenset[str]]],
        split_key: str,
        test_fraction: float = 0.5,
    ):
        covariate_counts = defaultdict(int)
        for _, covariates in heldout_perturbation_covariates:
            covariate_counts[covariates] += 1
        
        # Separate heldout items into those that will remain heldout vs those that go back to train
        heldout_to_keep = []
        heldout_to_train = []
        
        for perturbation, covariates in heldout_perturbation_covariates:
            if covariate_counts[covariates] > 1:
                heldout_to_keep.append((perturbation, covariates))
            else:
                heldout_to_train.append((perturbation, covariates))
        
        # Add the singleton heldout items back to train
        train_perturbation_covariates = list(train_perturbation_covariates) + heldout_to_train
        
        log.debug(
            "Covariate filtering: %d heldout items kept, %d moved back to train",
            len(heldout_to_keep),
            len(heldout_to_train)
        )
        log.debug(heldout_to_keep)
   
        validation_perturbation_covariates, test_perturbation_covariates = (
            train_test_split(
                heldout_to_keep,
                stratify=[
                    str(covariates) for _, covariates in heldout_to_keep
                ],
                test_size=test_fraction,  ## Split test and test perturbations evenly
                random_state=seed,
            )
        )

        train_perturbation_covariates = set(train_perturbation_covariates)
        validation_perturbation_covariates = set(validation_perturbation_covariates)
        test_perturbation_covariates = set(test_perturbation_covariates)

        self.obs_dataframe[split_key] = [None] * self.obs_dataframe.shape[0]
        self.obs_dataframe.loc[
            [x in train_perturbation_covariates for x in self.perturbation_covariates],
            split_key,
        ] = "train"
        self.obs_dataframe.loc[
            [
                x in validation_perturbation_covariates
                for x in self.perturbation_covariates
            ],
            split_key,
        ] = "val"
        self.obs_dataframe.loc[
            [x in test_perturbation_covariates for x in self.perturbation_covariates],
            split_key,
        ] = "test"

    def _split_controls(
            self,
            seed,
            split_key,
            train_control_fraction,
            valid_covariates=None,
        ):
        """
        Split control cells into train/val/test, requiring at least 1 control cell
        in each split *per covariate* (defined by self.covariate_keys).
        
        For covariate groups with fewer than 3 control cells, all controls are
        assigned to 'train' only.
        
        For covariate groups not in valid_covariates (i.e., those without perturbed
        cells), all controls are assigned to 'train' only.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        split_key : str
            Column name in obs_dataframe to store the split assignments
        train_control_fraction : float
            Fraction of control cells to assign to training set
        valid_covariates : set of frozenset, optional
            Set of valid covariate combinations (those with both controls and
            perturbed cells). Controls in covariates not in this set will be
            assigned to 'train' only.

        Raises
        ------
        ValueError
            If covariate_keys is not set on the object.
        """
        random.seed(seed)

        df = self.obs_dataframe

        # All control cells
        ctrl_mask = df[self.perturbation_key] == self.perturbation_control_value
        ctrl_df = df.loc[ctrl_mask]

        # Determine covariate columns
        covariate_cols = getattr(self, "covariate_keys", None)
        if covariate_cols is None:
            raise ValueError(
                "covariate_keys must be set on the object to split controls per covariate."
            )
        if isinstance(covariate_cols, str):
            covariate_cols = [covariate_cols]

        # Fractions for val/test given total control fraction
        val_control_frac = (1.0 - train_control_fraction) / 2.0

        # Group controls by covariate(s) and split within each group
        for cov_value, idx in ctrl_df.groupby(covariate_cols, observed=False).groups.items():
            cov_ix = list(idx)
            n_ctrl = len(cov_ix)
            
            # Check if this covariate is in the valid_covariates set
            cov_frozen = frozenset(cov_value) if not isinstance(cov_value, tuple) else frozenset(cov_value)
            is_valid_covariate = valid_covariates is None or cov_frozen in valid_covariates

            # If covariate is not valid (no perturbed cells), assign all controls to train
            if not is_valid_covariate:
                log.debug(
                    f"Covariate group {cov_value} is not in valid_covariates "
                    f"(no perturbed cells). Assigning all {n_ctrl} control cells to 'train'."
                )
                df.loc[cov_ix, split_key] = "train"
                continue

            # If fewer than 3 control cells, assign all to train
            if n_ctrl < 3:
                log.debug(
                    f"Covariate group {cov_value} has only {n_ctrl} control cells "
                    f"(< 3 required for train/val/test split). Assigning all to 'train'."
                )
                df.loc[cov_ix, split_key] = "train"
                continue

            # Shuffle indices within this covariate group
            cov_ix = random.sample(cov_ix, k=n_ctrl)

            # Initial allocation based on fractions
            n_train = int(round(train_control_fraction * n_ctrl))
            n_val = int(round(val_control_frac * n_ctrl))
            n_test = n_ctrl - n_train - n_val  # ensure sum matches exactly

            splits = {"train": n_train, "val": n_val, "test": n_test}

            # Enforce at least 1 per split by borrowing from the largest donors
            while min(splits.values()) < 1:
                # Split that needs at least one
                need = next(k for k, v in splits.items() if v < 1)

                # Donors with >1 cell
                donors = [k for k, v in splits.items() if v > 1]
                if not donors:
                    # With n_ctrl >= 3 this shouldn't happen, but keep a guard
                    raise RuntimeError(
                        f"Could not allocate at least one cell per split for covariate "
                        f"group {cov_value} (splits={splits}, n_ctrl={n_ctrl})."
                    )

                donor = max(donors, key=lambda k: splits[k])
                splits[need] += 1
                splits[donor] -= 1

            n_train = splits["train"]
            n_val = splits["val"]
            n_test = splits["test"]

            # Final sanity checks
            assert n_train >= 1 and n_val >= 1 and n_test >= 1
            assert n_train + n_val + n_test == n_ctrl

            train_ix, val_ix, test_ix = np.split(
                cov_ix,
                [n_train, n_train + n_val],
            )

            df.loc[train_ix, split_key] = "train"
            df.loc[val_ix, split_key] = "val"
            df.loc[test_ix, split_key] = "test"

        # write back (df is self.obs_dataframe view, but keep explicit)
        self.obs_dataframe = df
        
    def _summarize_split(self, split_key):
        unique_covariates_merged = [x for x in set(self.covariates_merged)]
        split_summary_df = pd.DataFrame(
            0,
            index=[str(tuple(x)) for x in unique_covariates_merged],
            columns=["train", "val", "test"],
        )
        for covariates in unique_covariates_merged:
            obs_df_sub = self.obs_dataframe.loc[
                [x == covariates for x in self.covariates_merged]
            ]
            train_perts = obs_df_sub.loc[
                obs_df_sub[split_key] == "train", self.perturbation_key
            ].unique()
            val_perts = obs_df_sub.loc[
                obs_df_sub[split_key] == "val", self.perturbation_key
            ].unique()
            test_perts = obs_df_sub.loc[
                obs_df_sub[split_key] == "test", self.perturbation_key
            ].unique()

            split_summary_df.loc[str(tuple(covariates)), "train"] = len(train_perts)
            split_summary_df.loc[str(tuple(covariates)), "val"] = len(val_perts)
            split_summary_df.loc[str(tuple(covariates)), "test"] = len(test_perts)

        return split_summary_df

    def _downsample_combinatorial_perturbations(
        self, seed: int, downsample_fraction: float
    ):
        unique_perturbations = list(self.obs_dataframe[self.perturbation_key].unique())
        num_sample = np.round(
            len(unique_perturbations) * downsample_fraction, 0
        ).astype(int)

        random.seed(seed)
        sampled_perturbations = random.sample(unique_perturbations, k=num_sample)
        sampled_single_perturbations = set(
            [
                x
                for x in sampled_perturbations
                if self.perturbation_combination_delimiter not in x
            ]
        )
        sampled_single_perturbations.add(self.perturbation_control_value)

        sampled_combo_perturbations = []
        for combo_pert in [
            x
            for x in sampled_perturbations
            if self.perturbation_combination_delimiter in x
        ]:
            single_pert_list = combo_pert.split(self.perturbation_combination_delimiter)
            if all([x in sampled_single_perturbations for x in single_pert_list]):
                sampled_combo_perturbations.append(combo_pert)

        sampled_perturbations = sampled_single_perturbations.union(
            sampled_combo_perturbations
        )
        return sampled_perturbations

    def split_covariates(
        self,
        print_split: bool = True,
        seed: int = 54,
        min_train_covariates: int = 1,
        max_heldout_covariates: int = 2,
        max_heldout_fraction_per_covariate: float = 0.3,
        max_heldout_perturbations_per_covariate: int = 200,
        train_control_fraction: float = 0.5,
        test_fraction: float = 0.5,
        downsample_fraction: float = 1.0,
        min_control_cells_per_covariate: int = 3,
    ):
        """Holds out perturbations in specific covariates to test the ability
             of a model to transfer perturbation effects to new covariates.

        Args
            print_split: Whether to print the split summary
            seed: Random seed for reproducibility
            min_train_covariates: Minimum number of covariates to include in the
              training set. Must be at least one.
            max_heldout_covariates: Maximum number of covariates to hold out for each
              perturbation. Must be at least one.
            max_heldout_fraction_per_cov: Maximum fraction of perturbations to
              hold out for each unique set of covariates
            test_fraction: Fraction of held out perturbations to include in the test
              vs val set
            train_control_fraction: Fraction of control cells to include in the
              training set
            min_control_cells_per_covariate: Minimum number of control cells required
              for a covariate to be eligible for holdout (default: 3)

        Returns
            split: Split of the data into train/val/test as a pd.Series
        """

        split_key = "transfer_split_seed" + str(seed)  ## Unique key for this split
        self.split_params[split_key] = {
            "min_train_covariates": min_train_covariates,
            "max_heldout_covariates": max_heldout_covariates,
            "max_heldout_fraction_per_cov": max_heldout_fraction_per_covariate,
            "train_control_fraction": train_control_fraction,
            "min_control_cells_per_covariate": min_control_cells_per_covariate,
        }

        # Identify valid covariates (those with sufficient control cells AND perturbed cells)
        control_mask = self.obs_dataframe[self.perturbation_key] == self.perturbation_control_value
        perturbed_mask = self.obs_dataframe[self.perturbation_key] != self.perturbation_control_value
        
        control_df = self.obs_dataframe[control_mask]
        perturbed_df = self.obs_dataframe[perturbed_mask]
        
        # Build set of covariate combinations that have perturbed cells
        perturbed_covariates = set()
        for cov_keys, _ in perturbed_df.groupby(self.covariate_keys):
            # Handle both single covariate and multiple covariates
            if isinstance(cov_keys, tuple):
                perturbed_covariates.add(frozenset(cov_keys))
            else:
                perturbed_covariates.add(frozenset([cov_keys]))
        
        log.debug(
            "Found %d unique covariate combinations with perturbed cells",
            len(perturbed_covariates)
        )
        
        valid_covariates = set()
        for cov_keys, df in control_df.groupby(self.covariate_keys):
            # Check if this covariate has enough control cells
            if len(df) >= min_control_cells_per_covariate:
                # Handle both single covariate and multiple covariates
                if isinstance(cov_keys, tuple):
                    cov_frozen = frozenset(cov_keys)
                else:
                    cov_frozen = frozenset([cov_keys])
                
                # Also check if this covariate has at least one perturbed cell
                if cov_frozen in perturbed_covariates:
                    valid_covariates.add(cov_frozen)
                else:
                    log.debug(
                        "Excluding covariate %r: has %d control cells but NO perturbed cells",
                        cov_keys,
                        len(df)
                    )
        #TODO: else statement redirecting all non-valid covariates to training
        #   else:
                
                    
        max_heldout_dict = {}  ## Maximum number of perturbations that can be held out for each unique set of covariates
        for cov_keys, df in self.obs_dataframe.groupby(self.covariate_keys):
            cov_frozen = frozenset(cov_keys)
            # Only include covariates that meet the control cell threshold
            if cov_frozen in valid_covariates:
                num_cov_perts = df[self.perturbation_key].nunique()
                max_heldout_dict[cov_frozen] = min(
                    max_heldout_fraction_per_covariate * num_cov_perts,
                    max_heldout_perturbations_per_covariate,
                )

        log.debug(
            "Covariates eligible for holdout: %d (after control cell filtering)",
            len(max_heldout_dict)
        )

        perturbation_covariates_dict = {}  ## Dictionary to store unique covariates for each perturbation
        for pert_key, df in self.obs_dataframe.groupby([self.perturbation_key]):
            pert_key = pert_key[0]
            if pert_key != self.perturbation_control_value:
                cov_key_df = df.loc[:, self.covariate_keys].drop_duplicates()
                # Only include valid covariates for this perturbation
                unique_pert_covs = [
                    frozenset(x) for x in cov_key_df.values 
                    if frozenset(x) in valid_covariates
                ]
                if unique_pert_covs:  # Only add if there are valid covariates
                    perturbation_covariates_dict[pert_key] = unique_pert_covs

        ## Sort by number of covariates
        perturbation_covariates_dict = dict(
            sorted(
                perturbation_covariates_dict.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )

        ## Downsample
        if downsample_fraction < 1.0:
            random.seed(seed)
            perturbations_keep = set(
                random.sample(
                    list(perturbation_covariates_dict.keys()),
                    int(downsample_fraction * len(perturbation_covariates_dict)),
                )
            )
            perturbation_covariates_dict = {
                k: v
                for k, v in perturbation_covariates_dict.items()
                if k in perturbations_keep
            }

        rng = np.random.RandomState(seed)
        seed_list = [
            rng.randint(100000) for i in range(0, len(perturbation_covariates_dict))
        ]

        num_total_covs = len(max_heldout_dict)
        num_heldout_dict = defaultdict(
            int
        )  ## Counter for number of perturbations held out for each unique set of covariates

        ## Iterate through each perturbation and choose a random subset of covariates to hold out for that perturbation
        train_perturbation_covariates = []
        heldout_perturbation_covariates = []

        log.debug(
            "BEGIN heldout selection: n_perturbations=%d | min_train_covariates=%r | "
            "max_heldout_covariates=%r | num_total_covs=%r | seed_list_len=%r",
            len(perturbation_covariates_dict),
            min_train_covariates,
            max_heldout_covariates,
            num_total_covs,
            (len(seed_list) if "seed_list" in locals() and seed_list is not None else None),
        )
        log.debug(
            "Initial dict sizes: num_heldout_dict=%r | max_heldout_dict=%r",
            (len(num_heldout_dict) if num_heldout_dict is not None else None),
            (len(max_heldout_dict) if max_heldout_dict is not None else None),
        )

        # A couple of example keys to sanity-check alignment between dicts (no logic impact)
        try:
            _ex_cov = next(iter(num_heldout_dict))
            log.debug(
                "Example covariates key from num_heldout_dict: %r | heldout=%r | max=%r",
                _ex_cov,
                num_heldout_dict.get(_ex_cov),
                max_heldout_dict.get(_ex_cov),
            )
        except StopIteration:
            log.debug("num_heldout_dict is empty (no covariates tracked yet).")

        for i, items in enumerate(perturbation_covariates_dict.items()):
            perturbation, covariate_list = items
            num_covariates = len(covariate_list)
            sampled_covariates = []

            log.debug(
                "[i=%d] perturbation=%r | num_covariates=%d | min_train_covariates=%r",
                i,
                perturbation,
                num_covariates,
                min_train_covariates,
            )
            if num_covariates == 0:
                log.debug("[i=%d] perturbation=%r has EMPTY covariate_list.", i, perturbation)

            if num_covariates > min_train_covariates:
                log.debug(
                    "[i=%d] perturbation=%r passes threshold: %d > %r",
                    i,
                    perturbation,
                    num_covariates,
                    min_train_covariates,
                )

                covariate_pool = [
                    covariates
                    for covariates in covariate_list
                    if num_heldout_dict[covariates] < max_heldout_dict[covariates]
                ]  ## Check if the maximum number of perturbations have been held out for this set of covariates

                log.debug(
                    "[i=%d] perturbation=%r | covariate_pool_size=%d (from covariate_list_size=%d)",
                    i,
                    perturbation,
                    len(covariate_pool),
                    len(covariate_list),
                )

                # Extra diagnostics to explain *why* pool might be empty (only runs under info)
                if log.isEnabledFor(10):
                    try:
                        _missing_in_num = [c for c in covariate_list if c not in num_heldout_dict]
                        _missing_in_max = [c for c in covariate_list if c not in max_heldout_dict]
                        log.debug(
                            "[i=%d] perturbation=%r | missing_keys: in_num_heldout_dict=%d | in_max_heldout_dict=%d",
                            i,
                            perturbation,
                            len(_missing_in_num),
                            len(_missing_in_max),
                        )
                    except Exception as e:
                        log.debug(
                            "[i=%d] perturbation=%r | error while checking missing keys: %r",
                            i,
                            perturbation,
                            e,
                        )

                    try:
                        _ineligible = [
                            c for c in covariate_list
                            if (c in num_heldout_dict and c in max_heldout_dict)
                            and not (num_heldout_dict[c] < max_heldout_dict[c])
                        ]
                        log.debug(
                            "[i=%d] perturbation=%r | ineligible_covariates_due_to_cap=%d",
                            i,
                            perturbation,
                            len(_ineligible),
                        )
                    except Exception as e:
                        log.debug(
                            "[i=%d] perturbation=%r | error while computing ineligible covariates: %r",
                            i,
                            perturbation,
                            e,
                        )

                if len(covariate_pool) > 0:
                    log.debug(
                        "[i=%d] perturbation=%r | seeding RNG with seed_list[%d]=%r",
                        i,
                        perturbation,
                        i,
                        seed_list[i],
                    )
                    random.seed(seed_list[i])

                    # fix so that it cannot be bigger than covariate pool size
                    num_sample_range = (
                        1,
                        min(
                            len(covariate_pool),
                            np.max(
                                [
                                    len(covariate_pool)
                                    - num_total_covs
                                    + max_heldout_covariates,
                                    1,
                                ]
                            ),
                        ),
                    )

                    num_sample = random.randint(num_sample_range[0], num_sample_range[1])

                    sampled_covariates = random.sample(
                        covariate_pool, num_sample
                    )  ## Sample a random subset of covariates to hold out

                    for covariates in sampled_covariates:
                        num_heldout_dict[covariates] += 1

                    _prev_len = len(heldout_perturbation_covariates)
                    heldout_perturbation_covariates.extend(
                        [
                            (perturbation, covariates)
                            for covariates in sampled_covariates
                        ]
                    )

            _train_prev_len = len(train_perturbation_covariates)
            train_perturbation_covariates.extend(
                [
                    (perturbation, covariates)
                    for covariates in covariate_list
                    if covariates not in sampled_covariates
                ]
            )
            log.debug(
                "[i=%d] perturbation=%r | train_perturbation_covariates grew: %d -> %d | sampled_covariates_size=%d",
                i,
                perturbation,
                _train_prev_len,
                len(train_perturbation_covariates),
                len(sampled_covariates),
            )

        log.debug(
            "END heldout selection: heldout_perturbation_covariates_size=%d | train_perturbation_covariates_size=%d",
            len(heldout_perturbation_covariates),
            len(train_perturbation_covariates),
        )

        log.debug(heldout_perturbation_covariates)

        ## Split held out perturbation/covariate pairs into val and test sets
        self._assign_split(
            seed,
            train_perturbation_covariates,
            heldout_perturbation_covariates,
            split_key,
            test_fraction=test_fraction,
        )

        ## Split control cells
        self._split_controls(seed, split_key, train_control_fraction, valid_covariates)

        ## Print split
        split_summary_df = self._summarize_split(split_key)
        self.summary_dataframes[split_key] = split_summary_df
        if print_split:
            print("Split summary: ")
            print(split_summary_df)

        split = self.obs_dataframe[split_key]

        return split

    def split_covariates_manual(
        self,
        covariates_holdout: list[frozenset[str]],
        print_split: bool = True,
        seed: int = 54,
        max_heldout_fraction_per_covariate: float = 0.7,
        train_control_fraction: float = 0.5,
        test_fraction: float = 0.5,
    ):
        """Holds out perturbations in specific covariates to test the ability
           of a model to transfer perturbation effects to new covariates.

        Args
            covariates_holdout: List of covariates to hold out. Each unique set
              of covariates should be a tuple/list/set of strings.
            print_split: Whether to print the split summary
            seed: Random seed for reproducibility
            max_heldout_fraction_per_cov: Maximum fraction of perturbations to
              hold out for each unique set of covariates
            test_fraction: Fraction of held out perturbations to include in the test
              vs val set
            train_control_fraction: Fraction of control cells to include in the
              training set

        Returns
            split: Split of the data into train/val/test as a pd.Series
        """
        covariates_holdout = [frozenset(x) for x in covariates_holdout]

        split_key = "transfer_split_seed" + str(seed)  ## Unique key for this split
        self.split_params[split_key] = {
            "covariates_holdout": covariates_holdout,
            "max_heldout_fraction_per_cov": max_heldout_fraction_per_covariate,
            "train_control_fraction": train_control_fraction,
        }

        rng = np.random.RandomState(seed)
        seed_list = [rng.randint(100000) for i in range(0, len(covariates_holdout))]

        train_perturbation_covariates = []
        heldout_perturbation_covariates = []
        for covs, df in self.obs_dataframe.groupby(self.covariate_keys):
            covs = frozenset(covs)
            covs_perts = [
                x
                for x in df[self.perturbation_key].unique()
                if x != self.perturbation_control_value
            ]

            if covs in covariates_holdout:
                random.seed(seed_list[covariates_holdout.index(covs)])
                heldout_perts = random.sample(
                    covs_perts,
                    int(max_heldout_fraction_per_covariate * len(covs_perts)),
                )
            else:
                heldout_perts = []

            heldout_perturbation_covariates.extend(
                [(perturbation, covs) for perturbation in heldout_perts]
            )
            train_perturbation_covariates.extend(
                [
                    (perturbation, covs)
                    for perturbation in covs_perts
                    if perturbation not in heldout_perts
                ]
            )

        ## Split held out perturbation/covariate pairs into val and test sets
        self._assign_split(
            seed,
            train_perturbation_covariates,
            heldout_perturbation_covariates,
            split_key,
            test_fraction=test_fraction,
        )

        ## Split control cells
        self._split_controls(seed, split_key, train_control_fraction)

        ## Print split
        split_summary_df = self._summarize_split(split_key)
        self.summary_dataframes[split_key] = split_summary_df
        if print_split:
            print("Split summary: ")
            print(split_summary_df)

        split = self.obs_dataframe[split_key]
        return split

    
def apply_toml_manual_split(
    obs: pd.DataFrame,
    toml_path: str | Path,
    *,
    dataset_key: str | None = None,
    split_col: str = "transfer_split_seed1",
    perturbation_col: str = "Assign",
    covariate_col: str = "predicted.subclass",
    perturbation_suffix: str | None = None,
    default_split: str | None = None,
) -> pd.Series:
    """
    Apply a manual train/val/test split to `obs` based on a TOML file.

    Expected TOML structure (example):

        [datasets]
        boli_ctx = "test_boli/processed_data/Boli_Perturb_CTX_edit_L6_hvg.h5ad"

        [training]
        boli_ctx = "train"

        [zeroshot]

        [fewshot."boli_ctx.L6 CT CTX"]
        val  = [ "Xpo7",]
        test = [ "Tbr1", "Satb2",]

    Semantics:
    - First, everything in this dataset is set to the default split (usually "train").
      That default is taken from [training].[dataset_key] if present, otherwise from
      `default_split` arg, otherwise "train".
    - Then, for each entry under [fewshot] (and [zeroshot], if used), we parse keys of
      the form "dataset.covariate". For entries with matching dataset_key, we set
      obs rows to the specified split ("train"/"val"/"test") where:
        - obs[covariate_col] == covariate
        - obs[perturbation_col] matches each listed perturbation plus `perturbation_suffix`
          (e.g. "Tbr1" -> "Tbr1_0" if suffix="_0").
    """
    toml_path = Path(toml_path)
    with toml_path.open("rb") as f:
        config = tomllib.load(f)

    # training_cfg = config.get("training", {})
    # # Priority: TOML training section > explicit default_split arg > "train"
    # default_split_value = training_cfg.get(dataset_key, default_split or "train")

    # Initialize everything to default (e.g. 'train')
    obs[split_col] = "train"

    def _apply_section(section_name: str):
        section_cfg = config.get(section_name, {})
        for key, splits in section_cfg.items():
            # key format assumed to be "dataset.covariate"
            ds_name, covariate_value = key.split(".", 1)

            # if ds_name != dataset_key:
            #     continue

            for split_name in ("train", "val", "test"):
                perts = splits.get(split_name)
                if not perts:
                    continue

                # Map "Tbr1" -> "Tbr1_0" if perturbation_suffix is set
                pert_values = [
                    p + perturbation_suffix if perturbation_suffix and not p.endswith(perturbation_suffix) else p
                    for p in perts
                ]

                mask = (
                    obs[perturbation_col].isin(pert_values)
                    & (obs[covariate_col] == covariate_value)
                )
                obs.loc[mask, split_col] = split_name

    # Fewshot section (your current example)
    _apply_section("fewshot")
    # Zeroshot could use the same structure/semantics later if you want
    _apply_section("zeroshot")

    return obs[split_col]