{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Bank Customer Churn Prediction using Artificial Neural Networks (ANN) (D.L-supervised learning)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing the libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "3RR8ch_kDeMN"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import tensorflow as tf\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "ecNoQ5gBDeO7"
   },
   "outputs": [],
   "source": [
    "data=pd.read_csv('D:\\FC practical work\\Machine Learning\\DEEP learning notes of udemy\\Machine Learning A-Z (Codes and Datasets)\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Python\\Churn_Modelling.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 487
    },
    "id": "SNGBERoADeTc",
    "outputId": "b902193e-bfd2-4bc0-e793-f727a1be298a"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>9996</td>\n",
       "      <td>15606229</td>\n",
       "      <td>Obijiaku</td>\n",
       "      <td>771</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>9997</td>\n",
       "      <td>15569892</td>\n",
       "      <td>Johnstone</td>\n",
       "      <td>516</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>9998</td>\n",
       "      <td>15584532</td>\n",
       "      <td>Liu</td>\n",
       "      <td>709</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>9999</td>\n",
       "      <td>15682355</td>\n",
       "      <td>Sabbatini</td>\n",
       "      <td>772</td>\n",
       "      <td>Germany</td>\n",
       "      <td>Male</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>10000</td>\n",
       "      <td>15628319</td>\n",
       "      <td>Walker</td>\n",
       "      <td>792</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 14 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      RowNumber  CustomerId    Surname  CreditScore Geography  Gender  Age  \\\n",
       "0             1    15634602   Hargrave          619    France  Female   42   \n",
       "1             2    15647311       Hill          608     Spain  Female   41   \n",
       "2             3    15619304       Onio          502    France  Female   42   \n",
       "3             4    15701354       Boni          699    France  Female   39   \n",
       "4             5    15737888   Mitchell          850     Spain  Female   43   \n",
       "...         ...         ...        ...          ...       ...     ...  ...   \n",
       "9995       9996    15606229   Obijiaku          771    France    Male   39   \n",
       "9996       9997    15569892  Johnstone          516    France    Male   35   \n",
       "9997       9998    15584532        Liu          709    France  Female   36   \n",
       "9998       9999    15682355  Sabbatini          772   Germany    Male   42   \n",
       "9999      10000    15628319     Walker          792    France  Female   28   \n",
       "\n",
       "      Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0          2       0.00              1          1               1   \n",
       "1          1   83807.86              1          0               1   \n",
       "2          8  159660.80              3          1               0   \n",
       "3          1       0.00              2          0               0   \n",
       "4          2  125510.82              1          1               1   \n",
       "...      ...        ...            ...        ...             ...   \n",
       "9995       5       0.00              2          1               0   \n",
       "9996      10   57369.61              1          1               1   \n",
       "9997       7       0.00              1          0               1   \n",
       "9998       3   75075.31              2          1               0   \n",
       "9999       4  130142.79              1          1               0   \n",
       "\n",
       "      EstimatedSalary  Exited  \n",
       "0           101348.88       1  \n",
       "1           112542.58       0  \n",
       "2           113931.57       1  \n",
       "3            93826.63       0  \n",
       "4            79084.10       0  \n",
       "...               ...     ...  \n",
       "9995         96270.64       0  \n",
       "9996        101699.77       0  \n",
       "9997         42085.58       1  \n",
       "9998         92888.52       1  \n",
       "9999         38190.78       0  \n",
       "\n",
       "[10000 rows x 14 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1 - Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "bF5SDj9zGH6F"
   },
   "outputs": [],
   "source": [
    "df=data.drop(['RowNumber', 'CustomerId','Surname'], axis=1)\n",
    "x=df.drop(['Exited'], axis=1)\n",
    "y=data['Exited']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "MnkSkbADDeWG",
    "outputId": "ad83d23a-1255-47c5-ff32-8695c60389d7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 14)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "bRDI-h83DeYS",
    "outputId": "a9ea5b6b-8bd6-48cd-8b45-50afc70a16a0"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',\n",
       "       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',\n",
       "       'IsActiveMember', 'EstimatedSalary', 'Exited'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "-CSMaYXDDeak",
    "outputId": "1d387176-aa2f-4d03-8416-6e24c556963c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber            int64\n",
       "CustomerId           int64\n",
       "Surname             object\n",
       "CreditScore          int64\n",
       "Geography           object\n",
       "Gender              object\n",
       "Age                  int64\n",
       "Tenure               int64\n",
       "Balance            float64\n",
       "NumOfProducts        int64\n",
       "HasCrCard            int64\n",
       "IsActiveMember       int64\n",
       "EstimatedSalary    float64\n",
       "Exited               int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vfxkQ5gODedF",
    "outputId": "b01c9b72-da6d-4abc-cad6-c8968dc2e9e7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber          0\n",
       "CustomerId         0\n",
       "Surname            0\n",
       "CreditScore        0\n",
       "Geography          0\n",
       "Gender             0\n",
       "Age                0\n",
       "Tenure             0\n",
       "Balance            0\n",
       "NumOfProducts      0\n",
       "HasCrCard          0\n",
       "IsActiveMember     0\n",
       "EstimatedSalary    0\n",
       "Exited             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Encoding categorical data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "Na8UtQwWDefG"
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import *\n",
    "le=LabelEncoder()\n",
    "y= le.fit_transform(y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 487
    },
    "id": "feS45ipsJSZm",
    "outputId": "6531de1a-8df1-4d1d-92c8-a8372138786c"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>619</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>608</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>502</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>699</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>850</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>771</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>516</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>709</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>772</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>792</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      CreditScore  Geography  Gender  Age  Tenure    Balance  NumOfProducts  \\\n",
       "0             619          0       0   42       2       0.00              1   \n",
       "1             608          2       0   41       1   83807.86              1   \n",
       "2             502          0       0   42       8  159660.80              3   \n",
       "3             699          0       0   39       1       0.00              2   \n",
       "4             850          2       0   43       2  125510.82              1   \n",
       "...           ...        ...     ...  ...     ...        ...            ...   \n",
       "9995          771          0       1   39       5       0.00              2   \n",
       "9996          516          0       1   35      10   57369.61              1   \n",
       "9997          709          0       0   36       7       0.00              1   \n",
       "9998          772          1       1   42       3   75075.31              2   \n",
       "9999          792          0       0   28       4  130142.79              1   \n",
       "\n",
       "      HasCrCard  IsActiveMember  EstimatedSalary  \n",
       "0             1               1        101348.88  \n",
       "1             0               1        112542.58  \n",
       "2             1               0        113931.57  \n",
       "3             0               0         93826.63  \n",
       "4             1               1         79084.10  \n",
       "...         ...             ...              ...  \n",
       "9995          1               0         96270.64  \n",
       "9996          1               1        101699.77  \n",
       "9997          0               1         42085.58  \n",
       "9998          1               0         92888.52  \n",
       "9999          1               0         38190.78  \n",
       "\n",
       "[10000 rows x 10 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x['Geography']=le.fit_transform(x['Geography'])\n",
    "x['Gender']=le.fit_transform(x['Gender'])\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "hsF3ERq_KB_O",
    "outputId": "b0958f31-989d-4410-f246-9a7ac614621b"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method Series.unique of 0        France\n",
       "1         Spain\n",
       "2        France\n",
       "3        France\n",
       "4         Spain\n",
       "         ...   \n",
       "9995     France\n",
       "9996     France\n",
       "9997     France\n",
       "9998    Germany\n",
       "9999     France\n",
       "Name: Geography, Length: 10000, dtype: object>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Geography'].unique"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 423
    },
    "id": "009IKg1MLPVV",
    "outputId": "43f34594-2dab-4d1b-a8e4-9e7fea81490c"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>619</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>608</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>502</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>699</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>850</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>771</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>516</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>709</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>772</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>792</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      CreditScore  Geography  Gender  Age  Tenure    Balance  NumOfProducts  \\\n",
       "0             619          0       0   42       2       0.00              1   \n",
       "1             608          2       0   41       1   83807.86              1   \n",
       "2             502          0       0   42       8  159660.80              3   \n",
       "3             699          0       0   39       1       0.00              2   \n",
       "4             850          2       0   43       2  125510.82              1   \n",
       "...           ...        ...     ...  ...     ...        ...            ...   \n",
       "9995          771          0       1   39       5       0.00              2   \n",
       "9996          516          0       1   35      10   57369.61              1   \n",
       "9997          709          0       0   36       7       0.00              1   \n",
       "9998          772          1       1   42       3   75075.31              2   \n",
       "9999          792          0       0   28       4  130142.79              1   \n",
       "\n",
       "      HasCrCard  IsActiveMember  EstimatedSalary  \n",
       "0             1               1        101348.88  \n",
       "1             0               1        112542.58  \n",
       "2             1               0        113931.57  \n",
       "3             0               0         93826.63  \n",
       "4             1               1         79084.10  \n",
       "...         ...             ...              ...  \n",
       "9995          1               0         96270.64  \n",
       "9996          1               1        101699.77  \n",
       "9997          0               1         42085.58  \n",
       "9998          1               0         92888.52  \n",
       "9999          1               0         38190.78  \n",
       "\n",
       "[10000 rows x 10 columns]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "cmA5VXWALXmV",
    "outputId": "47fe8e20-1ab3-40b0-e9f9-5552c5153308"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 1, ..., 1, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Splitting the dataset into the Training set and Test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "QcewnSjZDehb"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Feature Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "ztNPt3ZJLMKO"
   },
   "outputs": [],
   "source": [
    "sc=StandardScaler()\n",
    "x_train=sc.fit_transform(x_train)\n",
    "x_test=sc.fit_transform(x_test) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2 - Building the ANN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Initializing the ANN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "xOBSvGmiL6mt"
   },
   "outputs": [],
   "source": [
    "ann=tf.keras.models.Sequential()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Adding the input layer and the first hidden layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "id": "DXuv8AqlL6pR"
   },
   "outputs": [],
   "source": [
    "ann.add(tf.keras.layers.Dense(units=6, activation='relu'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Adding the second hidden layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "upPypRbBPn-d"
   },
   "outputs": [],
   "source": [
    "ann.add(tf.keras.layers.Dense(units=6, activation='relu'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "IJCYXa6iPtaW"
   },
   "outputs": [],
   "source": [
    "ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3 - Training the ANN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Compiling the ANN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "zftEtRjlP1Xl"
   },
   "outputs": [],
   "source": [
    "ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training the ANN on the Training set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "MGpYu8eDP8zD",
    "outputId": "087df5b3-1d8d-4e31-f8db-25dd4a894630"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "234/234 [==============================] - 2s 2ms/step - loss: 0.9871 - accuracy: 0.3313\n",
      "Epoch 2/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.6529 - accuracy: 0.7051\n",
      "Epoch 3/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.5923 - accuracy: 0.7963\n",
      "Epoch 4/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.5608 - accuracy: 0.7993\n",
      "Epoch 5/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.5367 - accuracy: 0.7997\n",
      "Epoch 6/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.5112 - accuracy: 0.8007\n",
      "Epoch 7/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.4783 - accuracy: 0.8089\n",
      "Epoch 8/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.4496 - accuracy: 0.8167\n",
      "Epoch 9/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.4342 - accuracy: 0.8206\n",
      "Epoch 10/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.4256 - accuracy: 0.8259\n",
      "Epoch 11/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.4189 - accuracy: 0.8271\n",
      "Epoch 12/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.4120 - accuracy: 0.8316\n",
      "Epoch 13/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.4046 - accuracy: 0.8336\n",
      "Epoch 14/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3947 - accuracy: 0.8414\n",
      "Epoch 15/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3816 - accuracy: 0.8466\n",
      "Epoch 16/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3718 - accuracy: 0.8491\n",
      "Epoch 17/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3664 - accuracy: 0.8517\n",
      "Epoch 18/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3632 - accuracy: 0.8524\n",
      "Epoch 19/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3609 - accuracy: 0.8526\n",
      "Epoch 20/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3596 - accuracy: 0.8521\n",
      "Epoch 21/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3583 - accuracy: 0.8537\n",
      "Epoch 22/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3574 - accuracy: 0.8541\n",
      "Epoch 23/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3560 - accuracy: 0.8526\n",
      "Epoch 24/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3552 - accuracy: 0.8546\n",
      "Epoch 25/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3548 - accuracy: 0.8559\n",
      "Epoch 26/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3542 - accuracy: 0.8569\n",
      "Epoch 27/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3535 - accuracy: 0.8566\n",
      "Epoch 28/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3527 - accuracy: 0.8569\n",
      "Epoch 29/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3530 - accuracy: 0.8567\n",
      "Epoch 30/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3520 - accuracy: 0.8576\n",
      "Epoch 31/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3521 - accuracy: 0.8581\n",
      "Epoch 32/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3516 - accuracy: 0.8584\n",
      "Epoch 33/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3511 - accuracy: 0.8577\n",
      "Epoch 34/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3511 - accuracy: 0.8567\n",
      "Epoch 35/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3509 - accuracy: 0.8587\n",
      "Epoch 36/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3501 - accuracy: 0.8611\n",
      "Epoch 37/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3500 - accuracy: 0.8581\n",
      "Epoch 38/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3498 - accuracy: 0.8586\n",
      "Epoch 39/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3499 - accuracy: 0.8586\n",
      "Epoch 40/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3489 - accuracy: 0.8603\n",
      "Epoch 41/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3491 - accuracy: 0.8571\n",
      "Epoch 42/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3483 - accuracy: 0.8606\n",
      "Epoch 43/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3482 - accuracy: 0.8600\n",
      "Epoch 44/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3480 - accuracy: 0.8587\n",
      "Epoch 45/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3479 - accuracy: 0.8596\n",
      "Epoch 46/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3478 - accuracy: 0.8597\n",
      "Epoch 47/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3479 - accuracy: 0.8596\n",
      "Epoch 48/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3471 - accuracy: 0.8604\n",
      "Epoch 49/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3476 - accuracy: 0.8591\n",
      "Epoch 50/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3470 - accuracy: 0.8606\n",
      "Epoch 51/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3470 - accuracy: 0.8601\n",
      "Epoch 52/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3466 - accuracy: 0.8610\n",
      "Epoch 53/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3461 - accuracy: 0.8613\n",
      "Epoch 54/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3463 - accuracy: 0.8591\n",
      "Epoch 55/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3462 - accuracy: 0.8590\n",
      "Epoch 56/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3460 - accuracy: 0.8611\n",
      "Epoch 57/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3456 - accuracy: 0.8603\n",
      "Epoch 58/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3459 - accuracy: 0.8597\n",
      "Epoch 59/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3454 - accuracy: 0.8589\n",
      "Epoch 60/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3452 - accuracy: 0.8593\n",
      "Epoch 61/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3458 - accuracy: 0.8596\n",
      "Epoch 62/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3445 - accuracy: 0.8623\n",
      "Epoch 63/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3447 - accuracy: 0.8587\n",
      "Epoch 64/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3448 - accuracy: 0.8596\n",
      "Epoch 65/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3447 - accuracy: 0.8596\n",
      "Epoch 66/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3451 - accuracy: 0.8600\n",
      "Epoch 67/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3447 - accuracy: 0.8599\n",
      "Epoch 68/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3445 - accuracy: 0.8593\n",
      "Epoch 69/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3445 - accuracy: 0.8600\n",
      "Epoch 70/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3448 - accuracy: 0.8579\n",
      "Epoch 71/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3447 - accuracy: 0.8599\n",
      "Epoch 72/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3447 - accuracy: 0.8586\n",
      "Epoch 73/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3442 - accuracy: 0.8604\n",
      "Epoch 74/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3446 - accuracy: 0.8593\n",
      "Epoch 75/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3443 - accuracy: 0.8589\n",
      "Epoch 76/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3447 - accuracy: 0.8583\n",
      "Epoch 77/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3443 - accuracy: 0.8586\n",
      "Epoch 78/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3442 - accuracy: 0.8590\n",
      "Epoch 79/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3440 - accuracy: 0.8604\n",
      "Epoch 80/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3442 - accuracy: 0.8581\n",
      "Epoch 81/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3440 - accuracy: 0.8580\n",
      "Epoch 82/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3442 - accuracy: 0.8601\n",
      "Epoch 83/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3440 - accuracy: 0.8596\n",
      "Epoch 84/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3440 - accuracy: 0.8593\n",
      "Epoch 85/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3439 - accuracy: 0.8599\n",
      "Epoch 86/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3436 - accuracy: 0.8591\n",
      "Epoch 87/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3439 - accuracy: 0.8589\n",
      "Epoch 88/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3438 - accuracy: 0.8589\n",
      "Epoch 89/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3437 - accuracy: 0.8583\n",
      "Epoch 90/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3439 - accuracy: 0.8583\n",
      "Epoch 91/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3433 - accuracy: 0.8594\n",
      "Epoch 92/100\n",
      "234/234 [==============================] - 0s 2ms/step - loss: 0.3433 - accuracy: 0.8587\n",
      "Epoch 93/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3437 - accuracy: 0.8587\n",
      "Epoch 94/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3432 - accuracy: 0.8579\n",
      "Epoch 95/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3441 - accuracy: 0.8581\n",
      "Epoch 96/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3438 - accuracy: 0.8587\n",
      "Epoch 97/100\n",
      "234/234 [==============================] - 1s 3ms/step - loss: 0.3434 - accuracy: 0.8589\n",
      "Epoch 98/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3429 - accuracy: 0.8590\n",
      "Epoch 99/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3427 - accuracy: 0.8601\n",
      "Epoch 100/100\n",
      "234/234 [==============================] - 1s 2ms/step - loss: 0.3433 - accuracy: 0.8576\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x2450ee0ef10>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ann.fit(x_train, y_train, batch_size = 30, epochs = 100)"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "# Part 4 - Making the predictions and evaluating the model\n",
    "Predicting the result of a single observation\n",
    "Homework\n",
    "\n",
    "Use our ANN model to predict if the customer with the following informations will leave the bank:\n",
    "\n",
    "Geography: France\n",
    "\n",
    "Credit Score: 600\n",
    "\n",
    "Gender: Male\n",
    "\n",
    "Age: 40 years old\n",
    "\n",
    "Tenure: 3 years\n",
    "\n",
    "Balance: $ 60000\n",
    "\n",
    "Number of Products: 2\n",
    "\n",
    "Does this customer have a credit card ? Yes\n",
    "\n",
    "Is this customer an Active Member: Yes\n",
    "\n",
    "Estimated Salary: $ 50000\n",
    "\n",
    "So, should we say goodbye to that customer ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "NmhHZCriQP9I",
    "outputId": "baee5f72-6ba6-4806-9003-4dd2307217f2"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 195ms/step\n",
      "[[ True]]\n"
     ]
    }
   ],
   "source": [
    "print(ann.predict(sc.transform([[0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "f0z9zCccQrQO",
    "outputId": "14530418-2a16-4d23-86c5-c0640b49ccff"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "94/94 [==============================] - 0s 2ms/step\n"
     ]
    }
   ],
   "source": [
    "y_pred = ann.predict(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Predicting the Test set results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "94/94 [==============================] - 0s 2ms/step\n",
      "[[0 0]\n",
      " [0 1]\n",
      " [0 0]\n",
      " ...\n",
      " [0 0]\n",
      " [0 0]\n",
      " [0 1]]\n"
     ]
    }
   ],
   "source": [
    "y_pred = ann.predict(x_test)\n",
    "y_pred = (y_pred > 0.5)\n",
    "print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Making the Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 363
    },
    "id": "9rpcdQB9QTUl",
    "outputId": "da1b7e40-0c02-43d6-a14c-2767c54e4ddc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[2307   72]\n",
      " [ 351  270]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.859"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, accuracy_score\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(cm)\n",
    "accuracy_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "# Bank Customer Churn Prediction using Artificial Neural Networks (ANN)\n",
    "\n",
    "## Introduction\n",
    "\n",
    "Customer churn, the phenomenon where customers discontinue their relationship with a business, is a significant concern in the banking industry. Predicting customer churn can help banks take proactive measures to retain valuable customers. This project aims to develop a predictive model using Artificial Neural Networks (ANN), a type of deep learning algorithm, to identify customers who are likely to churn.\n",
    "\n",
    "## Dataset\n",
    "\n",
    "The dataset used for training and evaluation consists of customer information obtained from a bank. It includes various attributes such as age, gender, account balance, transaction history, etc., along with a binary label indicating whether the customer has churned or not. The dataset is not included in this repository due to its size, but a sample dataset or your own dataset can be used to replicate the project.\n",
    "\n",
    "## Methodology\n",
    "\n",
    "The project follows the following steps to predict customer churn:\n",
    "\n",
    "1. Data Preprocessing: The raw dataset is preprocessed to handle missing values, categorical variables, and feature scaling. Data is divided into training and testing sets.\n",
    "\n",
    "2. Model Architecture: An ANN model is constructed using TensorFlow. The architecture includes an input layer, one or more hidden layers, and an output layer. Activation functions, dropout, and batch normalization techniques may be employed to enhance model performance.\n",
    "\n",
    "3. Model Training: The ANN model is trained using the preprocessed training dataset. The training process involves forward propagation, backpropagation, and weight optimization using gradient descent algorithms. The model is iteratively trained on the data to minimize the prediction error.\n",
    "\n",
    "4. Model Evaluation: The trained model is evaluated using the preprocessed testing dataset. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's performance in predicting customer churn.\n",
    "\n",
    "5. Prediction: The trained model can be used to make predictions on new, unseen customer data. The input data needs to be preprocessed in the same manner as the training data before being fed into the model.\n",
    "\n",
    "\n",
    "Conclusion\n",
    "\n",
    "The Bank Customer Churn Prediction project demonstrates the application of Artificial Neural Networks (ANN) for predicting customer churn in the banking industry. By training an ANN model on customer data, banks can proactively identify customers who are likely to churn and take appropriate measures to retain them. The project provides a foundation for further research and development in customer churn prediction using deep learning techniques."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
