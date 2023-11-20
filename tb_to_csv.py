import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import traceback

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_metadata(exp_name: str):
    # extract architecture and experiment name
    _, arch, exp_name = exp_name.split("_", 2)
    # remove /tensorboard from the experiment name
    exp_name = exp_name.split("/")[0]
    return arch, exp_name


# Extraction function
def tflog2pandas(path):
    """
    Function to automatically export tensorboard data to a csv file adapted from:
    https://stackoverflow.com/questions/49697634/tensorboard-export-csv-file-from-command-line
    """
    try:
        # get all tensorboard folders
        tb_folder_paths = [t[0] for t in os.walk(path) if t[0][-11:] == "tensorboard"]

        df_dict = {}
        # loop over experiment folders
        for p in tb_folder_paths:
            arch, exp_name = extract_metadata(p)
            if arch not in df_dict:
                df_dict[arch] = {}
            event_acc = EventAccumulator(p)
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]

            # loop over variables to extract
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))

                if tag not in df_dict[arch]:
                    df_dict[arch][tag] = pd.DataFrame(data={exp_name: values})

                else:
                    df_dict[arch][tag][exp_name] = values

    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return df_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--path", type=str, help="path to the multi-trial experiment folder."
    )
    parser.add_argument("--save", action="store_true", help="write csv files.")

    args = parser.parse_args()

    path = args.path
    df_dict = tflog2pandas(path)

    # save csv files
    if args.save:
        for k in df_dict.keys():
            print(f"save {k}.csv in {path}")
            df_dict[k].to_csv(os.path.join(path, f"{k.replace('/', '_')}.csv"))

    # plot figures
    # loop over tags
    for k in df_dict[list(df_dict)[0]].keys():
        figure = plt.figure(figsize=(10, 8))
        plt.title(f"{k}")

        # loop over architectures
        for arch in df_dict.keys():
            df_dict[arch][k][f"{arch} mean"] = df_dict[arch][k].mean(axis=1)

            # loop over experiment iterations
            for exp in df_dict[arch][k].columns:
                if "mean" not in exp:
                    plt.plot(
                        df_dict[arch][k].index,
                        df_dict[arch][k][exp],
                        alpha=0.5,
                        label=f"{' '.join(exp.split('_'))}"
                    )
                else:
                    plt.plot(
                        df_dict[arch][k].index, df_dict[arch][k][exp],
                        linestyle='dashed',
                        color='black',
                        label=f"mean {arch}"
                    )
        plt.legend()
        fig_name = f"tb_{k.replace('/', '_').lower()}.png"
        plt.savefig(path + "/" + fig_name)
        print(f"saved figure {fig_name} at {path}")
