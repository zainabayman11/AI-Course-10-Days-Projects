# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix


# # Importing the dataset
# dataset = pd.read_csv('Data.csv')
# X = dataset.iloc[:, :-1]
# y = dataset.iloc[:, -1]

# imputer = SimpleImputer(missing_values= np.NAN,strategy='')
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])

# x=pd.DataFrame(X)



import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1️⃣ تحميل البيانات
df = pd.read_csv("Data.csv")

# X[start_row:end_row, start_col:end_col]


# 2️⃣ استخراج البيانات المستقلة `X` والتابعة `y`
X = df.iloc[:, :-1]  # جميع الأعمدة عدا الأخير
# X = df.iloc[:, :-1].values  # بيشيل اسم الاعمده ويحط بيدلها ارقام

y = df.iloc[:, -1]   # العمود الأخير فقط

# X = df.drop(columns=['target_column_name'])  # حذف العمود المستهدف من الميزات
# y = df['target_column_name']  # استخراج العمود المستهدف
# y = df[['Target']]  # وضع اسم العمود داخل قائمة ليبقى DataFrame

# 3️⃣ معالجة القيم المفقودة
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X.iloc[:, 1:3] = imputer.fit_transform(X.iloc[:, 1:3])

# 4️⃣ تحويل البيانات النصية إلى أرقام (One-Hot Encoding)
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# 5️⃣ تطبيع البيانات العددية
scaler = StandardScaler()
X[:, 3:] = scaler.fit_transform(X[:, 3:])





# 6️⃣ تقسيم البيانات إلى `Train` و `Test`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ حفظ البيانات بعد المعالجة لاستخدامها لاحقًا
import pickle
with open("processed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
