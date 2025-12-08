from DatasetLoader import DatasetLoader
import Preprocessing as prep


def main():
    loader = DatasetLoader(
        dataset_1_path="data/set-2/train_net.csv",
        dataset_2_path="data/set-3/cores_iot.csv"
    )

    batch_size = 10

    df1 = loader.df1
    df1 = prep.drop_irrelevant_identifiers(df1)
    df1 = df1.dropna()
    df1 = prep.encode_protocol(df1)
    df1 = prep.convert_ip_columns(df1)
    df1 = prep.encode_attack_labels(df1)
    x1, y1 = prep.split_train_test(df1, ["ANOMALY"])
    x1 = prep.fill_missing(x1)
    x1 = prep.standardize(x1)
    train_eq_b_x1  = prep.create_equal_batchesa(x1, batch_size)
    train_eq_b_y1  = prep.create_equal_batchesa(y1, batch_size)

    df2 = loader.df2
    x2, y2 = prep.split_train_test(df2, [19])
    x2 = prep.fill_missing(x2)
    x2 = prep.standardize(x2)
    train_eq_b_x2 = prep.create_equal_batchesa(x2, batch_size)
    train_eq_b_y2 = prep.create_equal_batchesa(y2, batch_size)


    batch_size = 100
    # progressive_batches = prep.create_progressive_attack_batchesa(df1_clean, batch_size)


if __name__ == "__main__":
    main()