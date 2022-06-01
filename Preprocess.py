#Preprocess step include
# Train test split (20%)
# Standard Scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(self,data):

        target = 'class'
        X = data.drop(target, axis=1)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)







