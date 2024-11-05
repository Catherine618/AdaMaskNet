import numpy as np


def load_and_display_npy(file_path, show_data = False, num_rows=5):
    """
    加载 .npy 文件并显示其形状和前 num_rows 条数据。

    :param file_path: .npy 文件路径
    :param num_rows: 要显示的前N条数据
    """
    # 加载 .npy 文件
    data = np.load(file_path, allow_pickle=True)

    # 输出数据的形状
    print(f"Shape of the data: {data.shape}")

    # 根据数据的维度输出前 num_rows 条数据
    if show_data:
        if data.ndim == 1:
            print(f"First {num_rows} entries of the data:")
            print(data[:num_rows])
        elif data.ndim == 2:
            print(f"First {num_rows} rows of the data:")
            print(data[:num_rows, :])
        elif data.ndim == 3:
            print(f"First {num_rows} 2D slices of the data:")
            for i in range(min(num_rows, data.shape[0])):
                print(f"Slice {i + 1}:\n{data[i]}")
        else:
            print("Data has more than 3 dimensions. Displaying the first element:")
            print(data.flat[:num_rows])

if __name__ == '__main__':
    # 示例用法
    datasets = [#"OPPORTUNITY",
                #"PAMAP2",
                "UCI_HAR",
                "UCI_HAR_2",
                #"UniMiB_SHAR",
                #"USC_HAD",
                #"WISDM"
    ]

    for dataset_name in datasets:
        print(f"--------{dataset_name}----------")

        x_train_path = f'../{dataset_name}/x_train.npy'
        y_train_path = f'../{dataset_name}/y_train.npy'

        x_test_path = f'../{dataset_name}/x_test.npy'
        y_test_path = f'../{dataset_name}/y_test.npy'

        print("x_train Data:")
        load_and_display_npy(x_train_path)

        print("\ny_train Data:")
        load_and_display_npy(y_train_path, show_data=True)

        print("x_test Data:")
        load_and_display_npy(x_test_path)

        print("\ny_test Data:")
        load_and_display_npy(y_test_path)


