{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from matplotlib import style"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "style.use('fivethirtyeight')\n",
    "\n",
    "import sklearn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "test=pd.read_csv(r\"C:\\Users\\onlyp\\Documents\\SUBLIME_TEXT_SAVES\\test.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
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
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>300.000000</td>\n",
       "      <td>300.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>50.936667</td>\n",
       "      <td>51.205051</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>28.504286</td>\n",
       "      <td>29.071481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>-3.467884</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>27.000000</td>\n",
       "      <td>25.676502</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>53.000000</td>\n",
       "      <td>52.170557</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>73.000000</td>\n",
       "      <td>74.303007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>100.000000</td>\n",
       "      <td>105.591837</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                x           y\n",
       "count  300.000000  300.000000\n",
       "mean    50.936667   51.205051\n",
       "std     28.504286   29.071481\n",
       "min      0.000000   -3.467884\n",
       "25%     27.000000   25.676502\n",
       "50%     53.000000   52.170557\n",
       "75%     73.000000   74.303007\n",
       "max    100.000000  105.591837"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
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
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>300.000000</td>\n",
       "      <td>300.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>50.936667</td>\n",
       "      <td>51.205051</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>28.504286</td>\n",
       "      <td>29.071481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>-3.467884</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>27.000000</td>\n",
       "      <td>25.676502</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>53.000000</td>\n",
       "      <td>52.170557</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>73.000000</td>\n",
       "      <td>74.303007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>100.000000</td>\n",
       "      <td>105.591837</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                x           y\n",
       "count  300.000000  300.000000\n",
       "mean    50.936667   51.205051\n",
       "std     28.504286   29.071481\n",
       "min      0.000000   -3.467884\n",
       "25%     27.000000   25.676502\n",
       "50%     53.000000   52.170557\n",
       "75%     73.000000   74.303007\n",
       "max    100.000000  105.591837"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(300, 2)\n"
     ]
    }
   ],
   "source": [
    "print(test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "test = test.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(300, 2)\n"
     ]
    }
   ],
   "source": [
    "print(test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
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
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>300.000000</td>\n",
       "      <td>300.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>50.936667</td>\n",
       "      <td>51.205051</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>28.504286</td>\n",
       "      <td>29.071481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>-3.467884</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>27.000000</td>\n",
       "      <td>25.676502</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>53.000000</td>\n",
       "      <td>52.170557</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>73.000000</td>\n",
       "      <td>74.303007</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>100.000000</td>\n",
       "      <td>105.591837</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                x           y\n",
       "count  300.000000  300.000000\n",
       "mean    50.936667   51.205051\n",
       "std     28.504286   29.071481\n",
       "min      0.000000   -3.467884\n",
       "25%     27.000000   25.676502\n",
       "50%     53.000000   52.170557\n",
       "75%     73.000000   74.303007\n",
       "max    100.000000  105.591837"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = test[['x']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = test['y']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 58.13295189  20.53526857  96.7467888   65.24602711  67.27833431\n",
      "  24.59988298 100.81140322  59.14910549  60.16525909  13.42219335\n",
      "  74.39140954  11.38988614 100.81140322  55.08449108  52.03603027\n",
      "  42.89064784  35.77757262   8.34142533  69.31064152  35.77757262\n",
      "  78.45602395  71.34294873  67.27833431  14.43834695  84.55294557\n",
      "  18.50296136  34.76141901  19.51911497  62.1975663   59.14910549\n",
      "  56.10064468  37.80987982  63.2137199   75.40756314  73.37525593\n",
      "  60.16525909  22.56757577  10.37373254  32.72911181  32.72911181\n",
      "  89.63371358   4.27681092  73.37525593  38.82603343  93.698328\n",
      "  38.82603343  18.50296136  96.7467888   45.93910865   7.32527173\n",
      "  17.48680776  90.64986719  14.43834695  92.68217439  39.84218703\n",
      "  33.74526541  36.79372622  31.7129582   26.63219019   5.29296452\n",
      "  45.93910865  61.18141269  32.72911181   4.27681092  80.48833115\n",
      "  51.01987666  12.40603974  54.06833747  45.93910865  67.27833431\n",
      "   6.30911812  72.35910233  80.48833115  87.60140638   3.26065731\n",
      "  95.7306352   13.42219335  12.40603974  66.26218071  53.05218387\n",
      "  -0.8039571   87.60140638  91.66602079  30.6968046   97.76294241\n",
      "  64.2298735   54.06833747  -0.8039571   64.2298735   69.31064152\n",
      "  27.64834379  20.53526857  24.59988298  20.53526857  22.56757577\n",
      "  27.64834379  54.06833747  20.53526857  46.95526225  40.85834063\n",
      "  45.93910865  13.42219335  45.93910865  46.95526225  18.50296136\n",
      "  -0.8039571   95.7306352   27.64834379  23.58372938  55.08449108\n",
      "  41.87449423   0.21219651  89.63371358  82.52063836  15.45450055\n",
      "  29.680651    79.47217755  43.90680144  54.06833747  92.68217439]\n"
     ]
    }
   ],
   "source": [
    "print(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x298e7b1acc0>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(y_test, predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\stats\\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.\n",
      "  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x298e634f898>"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAagAAAEfCAYAAAAUfVINAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvIxREBQAAIABJREFUeJzt3XtclGXeP/DPPSeGo5wHDyCmiGiipoGVRmG1WW22lhb6/CKKlcraDvYUa/22w9rjo2jbQddHIzJ/m7vRRqm52emhRSPEUtI8IIoYpAwCImfmdP/+MIF7ZoABB+b0eb9evmquue6b64KZ+c51FhoaGkQQERE5GZmjC0BERGQNAxQRETklBigiInJKDFBEROSUGKCIiMgpMUAREZFTYoAiIiKnxABFREROyS0CVFlZmaOL4FCsP+vvqTy57oD7198tAhQREbkfBigiInJKDFBEROSUGKCIiMgp2RygsrOzER8fD41Gg6SkJBQWFvaYt7q6Gunp6bj66qsRHByMRx55xCLPe++9h7lz5yI6OhpRUVG444478N133w2sFkRE5HZsClB5eXnIzMzEsmXLUFBQgISEBCxYsACVlZVW83d0dCA4OBhPPvkkZsyYYTXPnj178Lvf/Q7btm3D119/jZiYGNx99904efLkwGtDRERuw6YAtX79eixatAipqamIjY1FVlYWNBoNcnJyrOYfPXo0Vq9ejcWLFyMoKMhqnrfffhtLlizBlClTEBMTg9deew1+fn746quvBl4bIiJyG30GKJ1Oh5KSEiQnJ0vSk5OTsXfvXrsVRKfTob29HYGBgXa7JxERuS5FXxnq6upgNBoRFhYmSQ8LC0NNTY3dCrJixQr4+flh7ty5drsnkTPZXNoyKPe9jlOdyE31GaAuEQRB8lgURYu0gdqwYQM2b96MTz75BAEBAb3m7WnltLuvqO4L6+/89dfWyAfnxhGuUf/B4sl1B1y//jExMT0+12eACgkJgVwut2gt1dbWWrSqBmLDhg149dVX8eGHH2L69Ol95rdWmbKysl4r6e5Yf9eov8Y0OC0o4IxL1H8wuMrffrC4e/377BxQqVSYOnUq8vPzJen5+flITEy8rB++bt06rFixAh988AGuueaay7oXERG5F5u6+JYuXYqMjAxMnz4diYmJyMnJQXV1NdLS0gAA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u/R0l5KSgg0bNkAURfz3f/83Nm/ejIaGBkyfPh1r1qzBxIkTHVBa+3LpAEVERO7LI9ZBERGR62GAIiIip8QARURETokBioiInBIDFBEROSUGKCIickoMUERE5JQYoIiIyCkxQBERkVNigCIiIqfEAEXkYPn5+QgMDMSOHTssnvv8888RGBiIXbt2OaBkRI7FAEXkYElJSRg5ciQ++OADi+dyc3MRGhqKm266yQElI3IsBigiB5PJZLj33nvxxRdfoKGhoTO9qakJn332GebPnw+FwqVPxiEaEAYoIieQkpICnU6Hjz/+uDNt+/btaG1txX333efAkhE5Do/bIHISN910E5RKZef5XfPmzcOZM2ewb98+B5eMyDHYgiJyEikpKSgqKsLp06dx9uxZ7N69G/fee6+ji0XkMAxQRE7i7rvvhkqlQm5uLj788EOIooiFCxc6ulhEDsMuPiInkpqaiiNHjkClUiEwMBA7d+50dJGIHIYtKCInkpKSgrKyMhw+fJjde+Tx2IIiciIGgwETJ05EY2MjSktLMWzYMEcXichh2IIiciIymQwKhQJz585lcCKPxwBF5ES++OILnDlzBikpKY4uCpHDsYuPyAl8//33OHLkCNasWQMfHx8UFhZCJuP3R/JsfAcQOYF33nkHTz31FAIDA7Fp0yYGJyKwBUVERE6KX9OIiMgpMUAREZFTYoAiIiKnxABFREROiQGKiIicEgMUERE5pf8PcXmC76yhhI8AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot((y_test-predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.5780213522034683\n"
     ]
    }
   ],
   "source": [
    "print(metrics.mean_absolute_error(y_test,predictions))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10.244897884021034\n"
     ]
    }
   ],
   "source": [
    "print(metrics.mean_squared_error(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3.200765202888371\n"
     ]
    }
   ],
   "source": [
    "print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))\n"
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
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
