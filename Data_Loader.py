import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


## train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class prep_data_loder:

    def data_loder(self, data, BATCH_SIZE):
        data.pop('Measurement Date')
        data['class'] = data['class'].astype('category')
        encode_map = {'Abnormal': 1, 'Normal': 0}

        data['class'].replace(encode_map, inplace=True)
        X = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_data = TestData(torch.FloatTensor(X_test))

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=1)

        return train_loader, test_loader, X_test, y_test