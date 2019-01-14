{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['sample_submission.csv', 'test.csv', 'train.csv']\n"
     ]
    }
   ],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # CSV file I/O (e.g. pd.read_csv)\n",
    "import os # reading the input files we have access to\n",
    "\n",
    "print(os.listdir(r'C:\\Users\\chosun\\Desktop\\AN\\taxi-1'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "key                   object\n",
       "fare_amount          float64\n",
       "pickup_datetime       object\n",
       "pickup_longitude     float64\n",
       "pickup_latitude      float64\n",
       "dropoff_longitude    float64\n",
       "dropoff_latitude     float64\n",
       "passenger_count        int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df =  pd.read_csv(r'C:\\Users\\chosun\\Desktop\\AN\\taxi-1/train.csv', nrows = 10_000_000)\n",
    "train_df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_travel_vector_features(df):\n",
    "    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()\n",
    "    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()\n",
    "\n",
    "add_travel_vector_features(train_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "key                    0\n",
      "fare_amount            0\n",
      "pickup_datetime        0\n",
      "pickup_longitude       0\n",
      "pickup_latitude        0\n",
      "dropoff_longitude     69\n",
      "dropoff_latitude      69\n",
      "passenger_count        0\n",
      "abs_diff_longitude    69\n",
      "abs_diff_latitude     69\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(train_df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Old size: 10000000\n",
      "New size: 9999931\n"
     ]
    }
   ],
   "source": [
    "print('Old size: %d' % len(train_df))\n",
    "train_df = train_df.dropna(how = 'any', axis = 'rows')\n",
    "print('New size: %d' % len(train_df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Old size: 9999931\n",
      "New size: 9979187\n"
     ]
    }
   ],
   "source": [
    "print('Old size: %d' % len(train_df))\n",
    "train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]\n",
    "print('New size: %d' % len(train_df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(9979187, 3)\n",
      "(9979187,)\n"
     ]
    }
   ],
   "source": [
    "def get_input_matrix(df):\n",
    "    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))\n",
    "\n",
    "train_X = get_input_matrix(train_df)\n",
    "train_y = np.array(train_df['fare_amount'])\n",
    "\n",
    "print(train_X.shape)\n",
    "print(train_y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[147.16176525  76.95503724   6.39545245]\n"
     ]
    }
   ],
   "source": [
    "(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)\n",
    "print(w)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[147.16176525  76.95503724   6.39545245]\n"
     ]
    }
   ],
   "source": [
    "w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)\n",
    "print(w_OLS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "key                   object\n",
       "pickup_datetime       object\n",
       "pickup_longitude     float64\n",
       "pickup_latitude      float64\n",
       "dropoff_longitude    float64\n",
       "dropoff_latitude     float64\n",
       "passenger_count        int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df = pd.read_csv(r'C:\\Users\\chosun\\Desktop\\AN\\taxi-1/test.csv')\n",
    "test_df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['.anaconda', '.android', '.AndroidStudio3.1', '.AndroidStudio3.2', '.conda', '.condarc', '.emulator_console_auth_token', '.gradle', '.idlerc', '.ipynb_checkpoints', '.ipython', '.jupyter', '.keras', '.matplotlib', '3D Objects', 'Anaconda3', 'AppData', 'Application Data', 'Contacts', 'Cookies', 'Desktop', 'Documents', 'Downloads', 'Favorites', 'IntelGraphicsProfiles', 'Links', 'Local Settings', 'MicrosoftEdgeBackups', 'Music', 'My Documents', 'NetHood', 'NTUSER.DAT', 'ntuser.dat.LOG1', 'ntuser.dat.LOG2', 'NTUSER.DAT{8ebe95f7-3dcb-11e8-a9d9-7cfe90913f50}.TM.blf', 'NTUSER.DAT{8ebe95f7-3dcb-11e8-a9d9-7cfe90913f50}.TMContainer00000000000000000001.regtrans-ms', 'NTUSER.DAT{8ebe95f7-3dcb-11e8-a9d9-7cfe90913f50}.TMContainer00000000000000000002.regtrans-ms', 'ntuser.ini', 'OneDrive', 'Oracle', 'Pictures', 'PrintHood', 'Recent', 'Saved Games', 'Searches', 'SendTo', 'source', 'submission.csv', 'Templates', 'Untitled.ipynb', 'Untitled3-Copy1.ipynb', 'Untitled3.ipynb', 'Videos', '시작 메뉴']\n"
     ]
    }
   ],
   "source": [
    "add_travel_vector_features(test_df)\n",
    "test_X = get_input_matrix(test_df)\n",
    "test_y_predictions = np.matmul(test_X, w).round(decimals = 2)\n",
    "\n",
    "submission = pd.DataFrame(\n",
    "    {'key': test_df.key, 'fare_amount': test_y_predictions},\n",
    "    columns = ['key', 'fare_amount'])\n",
    "submission.to_csv('submission.csv', index = False)\n",
    "\n",
    "print(os.listdir('.'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
