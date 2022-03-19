# TSF-task-1
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8e6d0771",
   "metadata": {},
   "source": [
    "# Ishant Gedam  |  Data Science  |  The Sparks Foundation "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f041b46",
   "metadata": {},
   "source": [
    "## Task-1: \n",
    "## Predict The Percentage Of A Student Based On The Number Of Study Hours"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb26d197",
   "metadata": {},
   "source": [
    "### Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d34c9510",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "1d7c2017",
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
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"http://bit.ly/w-data\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b26364e",
   "metadata": {},
   "source": [
    "### Exploring Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8d6ce15f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 528.0 bytes\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "672d3330",
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
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4c2c4d6",
   "metadata": {},
   "source": [
    "### Plotting Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "9d37cf0f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x200a1570940>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(x='Hours',data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "e6e8ccb6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(336.9714285714286, 0.5, 'Scores')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x576 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaUAAAGoCAYAAADmTPpwAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAf5UlEQVR4nO3df5Dc9X3f8ef7pDMH+hEb+fQDkCqrJiYRTWTnSh25UDeyMwp2DEkLmCYZ1WEqJeOxIOo0IelM3f6RqT1NGIcmk0E2SZTYJpKNPTBOqoEqcUIGm0RgxUYhDmO4YIzQCQVbQvjsg333j/2efFDptCfdd7+f3X0+Zm72drWrfaGR9sX38/nc5xOZiSRJJRhqOoAkSdMsJUlSMSwlSVIxLCVJUjEsJUlSMRY2HaBDLhGU1E+i6QCl8kpJklQMS0mSVAxLqVAXr15DRPTk18Wr1zT9xyepR0WP7OjQEyHnU0Rwwx0PNh3jrOzetpEe+XslNcU5pdPwSkmSVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVAxLSZJUjIVNB1AfGlpIRDSd4qxcdMlqvvH1p5qOcVYuXr2GZ57+etMx5mzB8Hm8PPWdpmOclV7++1IqS0nzr/USN9zxYNMpzsrubRubjnDWnnn66z35575728aezA29/felVA7fSZKKYSlJkophKUmSimEpSZKKYSlJkorR16vvenWJrBrUw8vZpX7Q16XUq0tkwaWmjXE5u9Qoh+8kScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFiMxsOsMZRcRe4PUdPv31wHM1xulUKTnALKdSSg4wy+mUkqWOHM9l5uZ5/j37Qk+U0lxExP7MHDPH95il3BxgltMpJUspOQaFw3eSpGJYSpKkYvRjKe1sOkCllBxgllMpJQeY5XRKyVJKjoHQd3NKkqTe1Y9XSpKkHmUpSZKKYSlJkophKUmSitETpbR58+YE/PLLL7/65atjffr5d1o9UUrPPVfCTiOS1H2D9vnXE6UkSRoMlpIkqRiWkiSpGJaSJKkYlpIkqRiWkiSpGJaSJKkYlpIkqRiWkiSpGJaSJKkYC5sOIEmlaLWS8aMnOHxskhVLR1i7bBFDQ9F0rIFiKUkS7ULae/BZduw5wORUi5HhIW67fgOb16+0mLrI4TtJAsaPnjhZSACTUy127DnA+NETDScbLJaSJAGHj02eLKRpk1MtJo5PNpSo7ctf/jIR0fjXxavXdOW/1+E7SQJWLB1hZHjoFcU0MjzE8iUjDaaCqakpbrjjwUYzAOzetrEr7+OVkiQBa5ct4rbrNzAy3P5YnJ5TWrtsUcPJBotXSpIEDA0Fm9ev5LLtVzJxfJLlS1x91wRLSZIqQ0PButHFrBtd3HSUgeXwnSSpGJaSJKkYlpIkqRiWkiSpGJaSJKkYlpIkqRiWkiSpGJaSJKkYlpIkqRiWkiSpGLWWUkTcHBGPRsTBiLileuzCiLg/Ih6vbl9XZwZJUu+orZQi4nLgPwFXAD8MvDsiLgVuBfZl5qXAvuq+JEm1Xin9APDFzHwxM18C/gL4KeAaYFf1nF3AtTVmkCT1kDpL6VHgqohYFhEXAFcDq4EVmXkIoLpdfqoXR8TWiNgfEfuPHDlSY0xJKsvMz7+ms3RbbaWUmY8BHwbuB/YCfwu8NIfX78zMscwcGx0drSmlJJVn5udf01m6rdaFDpl5Z2a+JTOvAv4JeBw4HBGrAKrbiTozSJJ6R92r75ZXt2uAnwbuAu4FtlRP2QLcU2cGSVLvqPvk2bsjYhkwBbw/M5+PiA8BeyLiJuAp4LqaM0jSvGq1kvGjJzh8bJIVSz02fT7VWkqZeeUpHjsKbKrzfSWpLq1Wsvfgs+zYc4DJqRYjw0Pcdv0GNq9faTHNA3d0kKQ5GD964mQhAUxOtdix5wDjR080nKw/WEqSNAeHj02eLKRpk1MtJo5PNpSov1hKkjQHK5aOMDL8yo/OkeEhli8ZaShRf7GUJGkO1i5bxG3XbzhZTNNzSmuXLWo4WX+oe/WdJPWVoaFg8/qVXLb9SiaOT7J8iavv5pOlJElzNDQUrBtdzLrRxU1H6TuWkqSu8+d8dDqWkqSu8ud8NBsXOkjqKn/OR7PxSklS7WYO13176uXT/pyPczSylCTV6tXDdTdveiMjw0OvKCZ/zkfTHL6TVKtXD9ft2f80N2+61J/z0Sl5pSSpVq/elufQtyb5wy/8I7vedwVJ+nM+egVLSVKtprflmVlMz7/4XUaXnOcckv4/Dt9JqpXb8mguvFKSVCu35dFcWEqSaue2POqUw3eSpGJYSpKkYlhKkqRiWEqSpGJYSpKkYlhKkqRiuCRckkoWwe5tG5tOwYLh87ryPpaSpJ40MKfXZnLDHQ82naJrxWgpSeo5nl7bv5xTktRzPL22f9VaShHxSxFxMCIejYi7ImIkIi6MiPsj4vHq9nV1ZpDUf159HAZ87/Ra9bbaSikiLga2A2OZeTmwAHgvcCuwLzMvBfZV9yWpY9PHYczk6bX9oe7hu4XA+RGxELgAeAa4BthV/fou4NqaM0jqMx6H0b9qW+iQmd+IiN8AngK+DdyXmfdFxIrMPFQ951BELK8rg6T+5HEY/au2Uqrmiq4B3gB8E/hURPzsHF6/FdgKsGbNmjoiSuph/XwcxszPv0FT5/DdO4AnM/NIZk4BnwE2AocjYhVAdTtxqhdn5s7MHMvMsdHR0RpjSlJZZn7+NZ2l2+ospaeAt0bEBRERwCbgMeBeYEv1nC3APTVmkCT1kDrnlB6KiE8DjwAvAV8CdgKLgT0RcRPt4rqurgySpN5S644OmflB4IOvevg7tK+aJEl6BXd0kCQVw1KSJBXDUpIkFcNSkiQVw1KSJBXDUpIkFcND/qQBNDCntqrnWErSgPHUVpXM4TtpwHhqq0pmKUkDxlNbVTKH76QBM31q68xi6rdTW50z611eKUkDpt9PbZ2eM7v69ge48aMPcfXtD7D34LO0Wtl0NHXAKyVpwPT7qa2nmzO7bPuVfXkgYL+xlKQB1M+nts42Z9aP/739xuE7SX1les5spn6bM+tnlpKkvtLvc2b9zuE7SX2l3+fM+p2lJKnv9POcWb9z+E6SVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVAxLSZJUDEtJklQMS0mSVIzaSiki3hQRB2Z8HYuIWyLiwoi4PyIer25fV1cGSVJvqa2UMvOrmbkhMzcAPwK8CHwWuBXYl5mXAvuq+5IkdW34bhPwtcz8R+AaYFf1+C7g2i5lkCQVrlul9F7grur7FZl5CKC6XX6qF0TE1ojYHxH7jxw50qWYktS8mZ9/TWfpttpLKSJeA7wH+NRcXpeZOzNzLDPHRkdH6wkn6Zy0WskTR17gC197jieOvECrlU1H6gszP/+aztJt3ThP6SeARzLzcHX/cESsysxDEbEKmOhCBknzrNVK9h58lh17DjA51Tp5wuvm9Ss9UE9nrRvDdzfyvaE7gHuBLdX3W4B7upBB0jwbP3riZCEBTE612LHnAONHTzScTL2s1iuliLgAeCewbcbDHwL2RMRNwFPAdXVmkFSPw8cmTxbStMmpFhPHJz3xdT5FsHvbxqZTcNElq7vyPrWWUma+CCx71WNHaa/Gk9TDViwdYWR46BXFNDI8xPIlIw2m6kOZ3HDHg7W+xe5tG8ksYz7QHR0knZW1yxZx2/UbGBluf4xMzymtXbao4WTqZd1Y6CCpDw0NBZvXr+Sy7VcycXyS5UtGWLtskYscdE4sJakQrVYyfvQEh49NsmJpb3zADw0F60YXO4ekeWMpSQVwebXU5pySVACXV0ttlpJUgNmWV0uDxFKSCjC9vHoml1drEFlKUgFcXi21udBBKoDLq6U2S0kqhMurJYfvJEkFsZQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScVw7zupi3rxyHOpmywlqUs88lw6M4fvpC7xyHPpzCwlqUs88lw6M0tJ6hKPPJfOzFKSusQjz6Uzc6GD1CUeeS6dWa2lFBGvBT4GXA4k8PPAV4HdwFpgHLg+M5+vM4c03852abdHnkuzq3v47reAvZl5GfDDwGPArcC+zLwU2Ffdl3rG9NLuq29/gBs/+hBX3/4Aew8+S6uVTUeTel5tpRQRS4GrgDsBMvO7mflN4BpgV/W0XcC1dWWQ6uDSbqk+dV4prQOOAL8fEV+KiI9FxCJgRWYeAqhul5/qxRGxNSL2R8T+I0eO1BhTmhuXdqtuMz//ms7SbR2VUkT884g4r/r+7RGxvZovms1C4C3A72bmm4ETzGGoLjN3ZuZYZo6Njo52+jKpdi7tVt1mfv41naXbOr1Suht4OSLeSHs47g3AJ8/wmqeBpzPzoer+p2mX1OGIWAVQ3U7MObXUIJd2S/XpdPVdKzNfioifAj6Smf87Ir402wsy89mI+HpEvCkzvwpsAv6u+toCfKi6vecc8ktd59JuqT6dltJURNxIu0R+snpsuIPXfQD4RES8BngCeB/tq7M9EXET8BRw3dwiS81zabdUj05L6X3ALwC/nplPRsQbgI+f6UWZeQA41Zjopo4TSpIGRkellJl/FxG/Aqyp7j9Je/hNkqR50+nqu58EDgB7q/sbIuLeGnNJkgZQp6vv/jtwBfBNODks94ZaEkmSBlanpfRSZn7rVY+5p4okaV51utDh0Yj4D8CCiLgU2A48WF8sSdIg6vRK6QPAeuA7tH9o9lvALTVlkiQNqDNeKUXEAuDezHwH8F/rjyRJGlRnvFLKzJeBFyPi+7qQR5I0wDqdU5oEvhIR99PeWBWAzNxeSypJ0kDqtJT+pPqSJHVTBLu3baz1LS66ZHWtv/9cdLqjw65q/7rvrx76amZO1RdLkgRAJjfccW6LnXdv20hmb/wUT0elFBFvp31K7DgQwOqI2JKZf1lbMknSwOl0+O43gR+vjqAgIr4fuAv4kbqCSZIGT6elNDxdSACZ+Q8R0cnRFVLXtVrJ+NETHD42yYqlnnUk9ZJOS2l/RNwJ/FF1/2eAh+uJJJ29VivZe/BZduw5wORU6+SpsJvXr7SYpB7Q6Y4OvwgcpL290M20T4/9hbpCSWdr/OiJk4UEMDnVYseeA4wfPXGGV0oqQadXSguB38rM2+DkLg/n1ZZKOkuHj02eLKRpk1MtJo5Pekqs1AM6vVLaB5w/4/75wP+d/zjSuVmxdISR4Vf+tR4ZHmL5kpGGEkmai05LaSQzX5i+U31/QT2RpLO3dtkibrt+w8limp5TWrtsUcPJJHWi0+G7ExHxlsx8BCAixoBv1xdLOjtDQ8Hm9Su5bPuVTByfZPkSV99JvaTTUroF+FREPEP7cL+LgBvqCiWdi6GhYN3oYueQpB406/BdRPzLiFiZmX8DXAbsBl4C9gJPdiGfJGmAnGlO6Q7gu9X3Pwr8GvA7wPPAzhpzSZIG0JmG7xZk5j9V398A7MzMu4G7I+JArckkSQPnTFdKCyJiurg2AX8249c6nY+SJKkjZyqWu4C/iIjnaK+2ewAgIt4IfKvmbJKkATNrKWXmr0fEPmAVcF9+70COIeADdYeTJA2WMw7BZeYXT/HYP3Tym0fEOHAceBl4KTPHIuJC2qv41tI+n+n6zHy+88iSpH7V6Y4O5+LfZuaGzByr7t8K7MvMS2lvX3RrFzJIknpAN0rp1a6hfYot1e21DWSQJBWo7lJK4L6IeDgitlaPrcjMQwDV7fKaM0iSekTdy7rflpnPRMRy4P6I+PtOX1iV2FaANWvW1JVPkooz8/Nv0NR6pZSZz1S3E8BngSuAwxGxCqC6nTjNa3dm5lhmjo2OjtYZU5KKMvPzr+ks3VZbKUXEoohYMv098OPAo8C9wJbqaVuAe+rKIEnqLXUO360APhsR0+/zyczcGxF/A+yJiJuAp4DraswgSeohtZVSZj4B/PApHj9Ke8siSZJeoYkl4ZIknZKlJEkqhqUkSSqGx09ooLVayfjRExw+NsmKpSOsXbaIoaFoOpY0sCwlDaxWK9l78Fl27DnA5FSLkeEhbrt+A5vXr7SYpIY4fKeBNX70xMlCApicarFjzwHGj55oOJk0uCwlDazDxyZPFtK0yakWE8cnG0okyVLSwFqxdISR4Vf+ExgZHmL5kpGGEkmylDSw1i5bxG3XbzhZTNNzSmuXLWo4mTS4XOiggTU0FGxev5LLtl/JxPFJli9x9Z3UNEtJA21oKFg3uph1o4ubjiIJh+8kSQWxlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFsJQkScWwlCRJxbCUJEnFcEcHSSpZBLu3bTy332NoIRHNb5910SWr+cbXn5r1OZaSPH1VKlkmN9zxYNMp5kUn5WopDThPX5VUEueUBpynr0oqiaU04Dx9VVJJHL4bcNOnr84splJPX3XuS+p/XikNuF45fXV67uvq2x/gxo8+xNW3P8Deg8/SamXT0STNI6+UBlyvnL56urmvy7Zf6QF9Uh+p/UopIhZExJci4nPV/Qsj4v6IeLy6fV3dGTS76dNX37ru9awbXVxcIYFzX9Kg6Mbw3c3AYzPu3wrsy8xLgX3VfWlW03NfM5U69yXp7NVaShFxCfAu4GMzHr4G2FV9vwu4ts4M6g+9Mvcl6dzUPaf0EeCXgSUzHluRmYcAMvNQRCw/1QsjYiuwFWDNmjU1x1TpemXuS5oPMz//Bk1tV0oR8W5gIjMfPpvXZ+bOzBzLzLHR0dF5Tqde1AtzX9J8mPn513SWbqvzSultwHsi4mpgBFgaER8HDkfEquoqaRUwUWMGSVIPqe1KKTN/NTMvycy1wHuBP8vMnwXuBbZUT9sC3FNXBklSb2nih2c/BLwzIh4H3lndlySpOz88m5mfBz5ffX8U2NSN95Uk9Ra3GZIkFcNSkiQVw1KSJBXDUpIkFcNSkiQVw1KSJBXDUpIkFcNSkiQVw1KSJBXDUpIkFaMr2wxpMLRayfjRExw+NsmKpZ53JGnuLCXNi1Yr2XvwWXbsOcDkVOvkybCb16+0mCR1zOE7zYvxoydOFhLA5FSLHXsOMH70RMPJJPUSS0nz4vCxyZOFNG1yqsXE8cmGEknqRZaS5sWKpSOMDL/yr9PI8BDLl4w0lEhSL7KUNC/WLlvEbddvOFlM03NKa5ctajiZpF7iQgfNi6GhYPP6lVy2/Uomjk+yfImr7yTNnaWkeTM0FKwbXcy60cVNR5HUoxy+kyQVw1KSJBXDUpIkFcNSkiQVw1KSJBXDUpIkFcNSkiQVw1KSJBXDUpIkFcNSkiQVo7ZthiJiBPhL4LzqfT6dmR+MiAuB3cBaYBy4PjOfrytHr5rtFNdunvDqabKSuqnOve++A/xYZr4QEcPAX0XE/wF+GtiXmR+KiFuBW4FfqTFHz5ntFFegaye8epqspG6rbfgu216o7g5XXwlcA+yqHt8FXFtXhl412ymu3Tzh1dNkJXVbrbuER8QC4GHgjcDvZOZDEbEiMw8BZOahiFh+mtduBbYCrFmzps6YxZntFNdMTvtr870792w53Alcqs/Mzz+A3ds2Nphm/iwYPu+Mz6m1lDLzZWBDRLwW+GxEXD6H1+4EdgKMjY1lPQnLNH2K68xCmHmK62y/1s0ckuox8/MvIvKGOx5sONH86KRcu7L6LjO/CXwe2AwcjohVANXtRDcy9JLZTnHt5gmvniYrqdsis56LkIgYBaYy85sRcT5wH/Bh4N8AR2csdLgwM395tt9rbGws9+/fX0vOUk2vejvVKa6z/Vo3c0g6ax3/I+q3K6Wqc07731/n8N0qYFc1rzQE7MnMz0XEF4A9EXET8BRwXY0ZetZsp7h284RXT5OV1E21lVJmfhl48ykePwpsqut9JUm9yx0dJEnFsJQkScWodUm4yuX2QZJKZCkNILcPklQqh+8GkNsHSSqVpTSAZts+SJKaZCkNoOntg2Zy+yBJJbCUBpDbB0kqlQsdBtDQULB5/Uou236l2wdJKoql1IPmYzm32wdJKpGl1GNczi2pnzmn1GNczi2pn1lKPcbl3JL6mcN386CbW/Z4GqykfuaV0jmanuO5+vYHuPGjD3H17Q+w9+CztFr1HJ7ocm5J/cwrpXN0ujmey7ZfWcvKNpdzS+pnltI5mm2Op67l1i7nltSvHL47R27ZI0nzx1I6R87xSNL8cfjuHDnHI0nzx1KaB87xSNL8cPhOklQMS0mSVAxLSZJUDOeUCtXNrYskqRSWUoE8nkLSoHL4rkAeTyFpUNVWShGxOiL+PCIei4iDEXFz9fiFEXF/RDxe3b6urgy9yuMpJA2qOq+UXgL+c2b+APBW4P0R8YPArcC+zLwU2FfdL0KrlTxx5AW+8LXneOLIC7Xt9H0mbl0kaVDVVkqZeSgzH6m+Pw48BlwMXAPsqp62C7i2rgxz0e0jKGbj1kWSBlVXFjpExFrgzcBDwIrMPATt4oqI5d3IcCbdPoJiNm5dJGlQ1V5KEbEYuBu4JTOPRXT2wRoRW4GtAGvWrKkvYKWJIyhm49ZF0uCa+fk3aGpdfRcRw7QL6ROZ+Znq4cMRsar69VXAxKlem5k7M3MsM8dGR0frjAk4jyOpHDM//5rO0m11rr4L4E7gscy8bcYv3Qtsqb7fAtxTV4a5cB5HkppX5/Dd24CfA74SEQeqx34N+BCwJyJuAp4CrqsxQ8ecx5Gk5tVWSpn5V8DpPtE31fW+58J5HElqljs6SJKKYSlJkophKUmSitGXu4R77IMk9aa+KyWPfZCk3tV3w3ce+yBJvavvrpRK2y5Iks7F8PAwu7dtbDrGvLjoktVnfE7fldL0dkEzi8ntgiT1qh/6oR9i//79Tcfomr4bvnO7IEnqXX13peR2QZLUu/qulMDtgiSpV/Xd8J0kqXdZSpKkYlhKkqRiWEqSpGJYSpKkYlhKkqRiWEqSpGJYSpKkYlhKkqRiRGY2neGMIuII8I8dPv31wHM1xulUKTnALKdSSg4wy+mUkqWOHM9l5uZOnhgRezt9bj/oiVKai4jYn5lj5vges5SbA8xyOqVkKSXHoHD4TpJUDEtJklSMfiylnU0HqJSSA8xyKqXkALOcTilZSskxEPpuTkmS1Lv68UpJktSjLCVJUjH6opQi4vciYiIiHi0gy+qI+POIeCwiDkbEzQ1mGYmIv46Iv62y/I+mslR5FkTElyLicw3nGI+Ir0TEgYjY33CW10bEpyPi76u/Mz/aUI43VX8e01/HIuKWhrL8UvX39dGIuCsiRprIUWW5ucpxsKk/j0HTF3NKEXEV8ALwh5l5ecNZVgGrMvORiFgCPAxcm5l/10CWABZl5gsRMQz8FXBzZn6x21mqPDuAMWBpZr67iQxVjnFgLDMb/8HMiNgFPJCZH4uI1wAXZOY3G860APgG8K8ys9MfWp+v976Y9t/TH8zMb0fEHuBPM/MPupmjynI58MfAFcB3gb3AL2bm493OMkj64kopM/8S+KemcwBk5qHMfKT6/jjwGHBxQ1kyM1+o7g5XX438X0hEXAK8C/hYE+9foohYClwF3AmQmd9tupAqm4CvdbuQZlgInB8RC4ELgGcayvEDwBcz88XMfAn4C+CnGsoyMPqilEoVEWuBNwMPNZhhQUQcACaA+zOzqSwfAX4ZaDX0/jMlcF9EPBwRWxvMsQ44Avx+Naz5sYhY1GCeae8F7mrijTPzG8BvAE8Bh4BvZeZ9TWQBHgWuiohlEXEBcDWwuqEsA8NSqklELAbuBm7JzGNN5cjMlzNzA3AJcEU1JNFVEfFuYCIzH+72e5/G2zLzLcBPAO+vhn+bsBB4C/C7mflm4ARwa0NZAKiGEN8DfKqh938dcA3wBuAiYFFE/GwTWTLzMeDDwP20h+7+FnipiSyDxFKqQTV/czfwicz8TNN5AKphoc8DTWzs+DbgPdVczh8DPxYRH28gBwCZ+Ux1OwF8lvacQROeBp6ecfX6adol1aSfAB7JzMMNvf87gCcz80hmTgGfATY2lIXMvDMz35KZV9GeInA+qWaW0jyrFhfcCTyWmbc1nGU0Il5bfX8+7X/wf9/tHJn5q5l5SWaupT009GeZ2cj//UbEomoBCtVQ2Y/THqbpusx8Fvh6RLypemgT0PUFMa9yIw0N3VWeAt4aERdU/5Y20Z6XbURELK9u1wA/TbN/NgNhYdMB5kNE3AW8HXh9RDwNfDAz72woztuAnwO+Us3lAPxaZv5pA1lWAbuq1VRDwJ7MbHQ5dgFWAJ9tf96xEPhkZu5tMM8HgE9Uw2ZPAO9rKkg1b/JOYFtTGTLzoYj4NPAI7aGyL9HsNj93R8QyYAp4f2Y+32CWgdAXS8IlSf3B4TtJUjEsJUlSMSwlSVIxLCVJUjEsJUlSMSwl9aWIeOFV9/9jRPx2U3kkdcZSkuag+pkvSTWxlDRwIuKfRcS+iPhydbumevwPIuLfz3jeC9Xt26szsj5J+4eiF0XEn1TnVD0aETc09J8i9Z2+2NFBOoXzZ+yoAXAhcG/1/W/TPntrV0T8PHA7cO0Zfr8rgMsz88mI+HfAM5n5LoCI+L55TS4NMK+U1K++nZkbpr+A/zbj134U+GT1/R8B/7qD3++vM/PJ6vuvAO+IiA9HxJWZ+a15Sy0NOEtJ+t7Bhy9R/ZuoNgN9zYznnDj55Mx/AH6Edjn9z4iYWXiSzoGlpEH0IO3dygF+hvbx2wDjtMsG2mf6DJ/qxRFxEfBiZn6c9oF0TR83IfUN55Q0iLYDvxcR/4X2ya/TO3N/FLgnIv4a2MeMq6NX+RfA/4qIFu3do3+x5rzSwHCXcElSMRy+kyQVw1KSJBXDUpIkFcNSkiQVw1KSJBXDUpIkFcNSkiQV4/8BEZmA0wJ1ZYIAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x432 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,8))\n",
    "sns.jointplot(x='Hours', y='Scores',data=df)\n",
    "plt.xlabel(\"Hours\")\n",
    "plt.ylabel(\"Scores\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "54c69aae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(3.674999999999997, 0.5, 'Scores')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 864x576 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,8))\n",
    "sns.lmplot(x='Hours', y='Scores',data=df,palette= 'Rainbow')\n",
    "plt.xlabel(\"Hours\")\n",
    "plt.ylabel(\"Scores\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c44abb28",
   "metadata": {},
   "source": [
    "### Splitting Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "58fcabc2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "b40e0209",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.iloc[:, :-1].values  \n",
    "y = df.iloc[:, 1].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5f825084",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b678b7ae",
   "metadata": {},
   "source": [
    "### Linear Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ade58cde",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "lm = LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1cbd4711",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lm.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "633509e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = lm.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f602e1e",
   "metadata": {},
   "source": [
    "### Evaluating the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "4adfbdec",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE: 4.183859899002975\n",
      "MSE: 21.5987693072174\n",
      "RMSE: 4.6474476121003665\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "\n",
    "print('MAE:', metrics.mean_absolute_error(y_test, pred))\n",
    "print('MSE:', metrics.mean_squared_error(y_test, pred))\n",
    "print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b73f987",
   "metadata": {},
   "source": [
    "### Ploting To Explore Predicted Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "e8db9d11",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Aniket\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2557: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Density'>"
      ]
     },
     "execution_count": 29,
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
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(y_test-pred,bins=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "734bf52b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "line = lm.coef_*X+lm.intercept_\n",
    "\n",
    "# Plotting for the test data\n",
    "plt.scatter(X, y)\n",
    "plt.plot(X, line);\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f6445159",
   "metadata": {},
   "source": [
    "### Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "e256a814",
   "metadata": {},
   "outputs": [],
   "source": [
    "hour = [9.25]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "73a1214c",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_pred = lm.predict([hour])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "877b9f4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "results = pd.DataFrame({\"Hours\":hour, \"Predicted Scores\":new_pred})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bea2ac5",
   "metadata": {},
   "source": [
    "### Final Result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "4053f329",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Hour Studied:  [9.25] \n",
      "Predicted Score:  [93.69173249]\n"
     ]
    }
   ],
   "source": [
    "results\n",
    "\n",
    "print(\"\\nHour Studied: \",hour,\"\\nPredicted Score: \",new_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf270ee1",
   "metadata": {},
   "source": [
    "# Thank You "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "392b15ad",
   "metadata": {},
   "source": [
    "## Pls Share Your Feedbacks And Suggestions To Make It Better."
   ]
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
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
