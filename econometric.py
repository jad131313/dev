import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm

#load excel
df = pd.read_excel("data_life_exp.xlsx") #file in repo folder

#we add labels to the variables
labels = {
    "fertility": "Fertility rate, total (birth per women)",
    "gdp": "Gross domestic product (constant 2015 US$)",
    "gdp_pc": "gdp per capita (constant 2015 US$)",
    "health_exp": "current health expenditure (pourcentage of gdp)",
    "life_expectancy": "Life expectacy at birth, total"
}

#I descriptives
#a list i want to analyze
cols = ["fertility", "gdp", "gdp_pc", "health_exp", "life_expectancy"]
# df for DataFrame from pandas, describe() to generate a statistical summary in df or DataFrame, like stata summ
stats = df[cols].describe().T[["count", "mean", "std", "min", "max"]]
print(stats) 

#II Graphes

#choosing fertility since its the first label in our list, and dropna to remove all the missing values from our df
x = df["fertility"].dropna()

#plt.hist to draw the histogram, bins is the number of bars in the hist(20 bars), and the density=true to make the curve align properly
plt.hist(x, bins=20, density=True)

#average fertility and standard deviation; x is a column and mean is the "moyenne" ≠ from "mediane" which is median; std is standard deviation to spreead out the valuees around the mean 
mu, sigma = x.mean(), x.std(ddof=1)

#createing x-values for the curve
xx = np.linspace(x.min(), x.max(), 300)

#Drawing the 2D curve 
plt.plot(xx, norm.pdf(xx, mu, sigma))

#Adding a title to the graph and displaying it
plt.title("histogram of fertility and fitted normal curve")
plt.show()

#scatter: life_expectancy vs fertility
plt.figure()
plt.scatter(df["fertility"], df["life_expectancy"])
plt.title("Life Expectancy vs Fertility Rate in 2022 ")
plt.xlabel("Fertility rate")
plt.ylabel("Life expectancy")
plt.show()

#We want the list of countries, with there value for life expectancy and fertility when life expectancy is less than 20
#we used df.loc to access the groupe of rows and columns by labels
print(df.loc[df["life_expectancy"] < 20, ["Country", "life_expectancy", "fertility"]])

#I can redo my graph with omitting that country
df_no_outlier = df[df["life_expectancy"] > 20]

plt.figure()
plt.scatter(df_no_outlier["fertility"], df_no_outlier["life_expectancy"])
plt.title("Life Expectancy vs Fertility Rate in 2022 (without outlier)")
plt.xlabel("Fertility rate")
plt.ylabel("Life expectancy")
plt.show()



#IIISimple linear regression model

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

#a) Régression simple: life_expectancy sur fertility
df_reg = df[["life_expectancy", "fertility"]].dropna().copy()

X = sm.add_constant(df_reg["fertility"]) #x cest la variable expliative
y = df_reg["life_expectancy"] #y cest la variable expliquee

model = sm.OLS(y, X).fit() #OLS is MCO in french, Moindres Carrés Ordinaires and .fit() to estimate the coefficients of alpha and beta
print(model.summary()) #to get a regression table like in stata

#calculating R^2 using two methodes, first, ESS/TSS and second, 1-RSS/TSS
ESS = model.ess
TSS = model.centered_tss
RSS = model.ssr
print("R2 (ESS/TSS) =", ESS/TSS)
print("R2 (1-RSS/TSS) =", 1-RSS/TSS)

#predict ychapeau
df_reg["ychapeau"] = model.predict(X)

#manually ychapeau
alpha_chapeau = model.params["const"]
beta_chapeau = model.params["fertility"]
df_reg["ychapeau_manually"] = alpha_chapeau + beta_chapeau * df_reg["fertility"]

#verifications simular to stata

#option 1: stata sum ychapeau and manually
#descriptives: check both columns have the same summary stats
print("\n--- option 1: descriptives")
print(df_reg[["ychapeau", "ychapeau_manually"]].describe())

#option 2: stata correlate ychapeau ychapeau manually
#correlation should be ~1 idf the two series are the same
print("\n--- option 2: correlation (should be ~1) ---")
print(df_reg["ychapeau"].corr(df_reg["ychapeau_manually"]))

#option 3: stata gen verif = 1 if equal
#use a tolerance because floating numbers can differ by tiny rounding errors
print("\n--- option 3: equality check with tolerance ---")
df_reg["verif"] = np.isclose(df_reg["ychapeau"], df_reg["ychapeau_manually"], atol=1e-9).astype(int) #atol is absolute tolerance to tell np.isclose how much tiny diference we are willing to accept and still say those two numbers are equal
print(df_reg["verif"].value_counts(),)
# Same as Stata: drop verif ychapeau_a_la_main
df_reg.drop(columns=["verif", "ychapeau_manually"], inplace=True, errors="ignore")


#Residuel (uchapeau) and histogram like stata

#1) residuel ychapeau = y - ychapeay
#stata equivalent: predict ychapeau, residual
df_reg["uchapeau"] = model.resid

#check residual mean ~ 0 (sum uchapeau)
print("\n--- Residuals: mean should be ~ 0 ---")
print(df_reg["uchapeau"].describe())

#2)residuels (manually) uchapeau = life expectancy - ychapeau
df_reg["uchapeau_manually"] = df_reg["life_expectancy"] - df_reg["ychapeau"]

#verify they match: stata corr uchapeau uchapeau_manually
print("\n--- residual check (corr should be ~ 1) ---")
print(df_reg["uchapeau"].corr(df_reg["uchapeau_manually"]))

#drop the manual column, in stata its drop uchapean_manually
df_reg.drop(columns=["uchapeau_manually"], inplace=True)

#3) histogram of residuels and the fitted normal curve
plt.figure()
plt.hist(df_reg["uchapeau"], bins=20, density=True)

mu_u = df_reg["uchapeau"].mean() #mean=moyenne
sigma_u = df_reg["uchapeau"].std(ddof=1) #std=ecart-type and ddof delta degree of freedon or degree de liberte
xx_u = np.linspace(df_reg["uchapeau"].min(), df_reg["uchapeau"].max(), 300)

plt.plot(xx_u, norm.pdf(xx_u, mu_u, sigma_u))
plt.title("Residuals (uchapeau): histogram and fitted normal curve")
plt.xlabel("Residual")
plt.ylabel("density")
plt.show()

#Homoskedasticity vs Heteroskedasticity (like Stata hettest)

from statsmodels.stats.diagnostic import het_breuschpagan

#before doing the statistical inference, we must ensure the reliability of our standard deviations
#thus, we need to test for heteroscedasticity
#in pyhton we dont need to re-run the model like stata
#hettest in python is Breusch-Pagan
#H0: homoskedasticity
#H1: heteroskedasticity
bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model.resid, model.model.exog)

print("\n--- Breusch-Pagan test (Stata: hettest) ---")
print("LM statistic =", bp_lm)
print("LM p-value   =", bp_lm_pvalue)
print("F statistic  =", bp_f)
print("F p-value    =", bp_f_pvalue)

# If p-value is very small (e.g., < 0.01), reject H0 -> heteroskedasticity
plt.figure()
plt.scatter(df_reg["fertility"], df_reg["uchapeau"])
plt.axhline(0)  # Stata: yline(0)
plt.title("Residuals vs Fertility (2022)")
plt.xlabel("Fertility")
plt.ylabel("Residuals (uchapeau)")
plt.show()

#Robust
robust_model = model.get_robustcov_results(cov_type="HC1") #heteroskedasticity-consistent estimator with small-sample correction

print("\n--- Regression with robust SE (Stata: robust) ---") #SE: Standard Error or écart-type estimé du coefficient
print(robust_model.summary())

#Statistical inference
import numpy as np
from scipy import stats

# We use the robust results (equivalent to Stata: regress ..., robust)
# robust_model = model.get_robustcov_results(cov_type="HC1")   # we already have this line

#Hypothesis test: we want: H0: beta = -6  vs H1: beta != -6

beta0 = -6  # value to test against

# Extract estimated coefficient and its robust standard error
import pandas as pd

names = robust_model.model.exog_names
params = pd.Series(robust_model.params, index=names)
bse    = pd.Series(robust_model.bse, index=names)

b_chapeau = params["fertility"]
se_chapeau = bse["fertility"]


# Degrees of freedom: df_deg = n - k
# n = number of observations used in regression
# k = number of parameters estimated (const + fertility = 2)
n = int(robust_model.nobs)
k = int(robust_model.df_model) + 1             # df_model excludes const, so +1
df_deg = n - k

print("df =", df_deg)

t_stat = (b_chapeau - beta0) / se_chapeau

# Two-sided p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_deg))

print("\n--- Test H0: beta = -6 (robust) ---")
print("b_hat =", b_chapeau)
print("robust SE =", se_chapeau)
print("t-stat =", t_stat)
print("p-value =", p_value)
print("df =", df)

# Interpretation:
#  if p-value < 0.05 -> reject H0 at 5%
#  if p-value < 0.10 -> reject H0 at 10%

#quantiles
# Normal critical value (two-sided 5% -> 0.975 quantile)
z_975 = stats.norm.ppf(0.975)

# Student t critical value (two-sided 5% -> 0.975 quantile)
t_975 = stats.t.ppf(0.975, df=df_deg)

print("\n--- Critical values ---")
print("z_0.975 =", z_975)      # ~ 1.96
print("t_0.975 (df) =", t_975)

#Confidence interval for beta
# 95% confidence interval for beta (robust)
ci_low  = b_chapeau - t_975 * se_chapeau
ci_high = b_chapeau + t_975 * se_chapeau

print("\n--- 95% CI for beta (robust) ---")
print("[", ci_low, ",", ci_high, "]")


#Multiple linear regression model

#I multiple regression

#we assume that fertility is not the only variable that can explain life expectancy
#therefore, government healthcare spending and GDP per capita are added
#Keep only the variables we need, and drop missing values

df_multi = df[["life_expectancy", "fertility", "health_exp", "gdp_pc"]].dropna().copy()

# we define y and X
y = df_multi["life_expectancy"]
X = sm.add_constant(df_multi[["fertility", "health_exp", "gdp_pc"]])

# we run OLS regression
model_multi = sm.OLS(y, X).fit()

#Display regression table inclusing R² and adjusted R² like Stata
print(model_multi.summary())

#print just R² and adjusted R² explicitly
print("\nR² =", model_multi.rsquared)
print("Adjusted R² =", model_multi.rsquared_adj)

#II Homoskedasticity vs Heteroskedasticity

from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

#full multiple regression: y on fertility + health_exp + gdp_pc
df_multi = df[["life_expectancy", "fertility", "health_exp", "gdp_pc"]].dropna().copy()

y = df_multi["life_expectancy"]
X = sm.add_constant(df_multi[["fertility", "health_exp", "gdp_pc"]])

model_multi = sm.OLS(y, X).fit()
print(model_multi.summary())

#Breusch–Pagan test (hettest)
#H0: homoskedasticity (constant error variance)
 #H1: heteroskedasticity
bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(model_multi.resid, model_multi.model.exog)

print("\n--- Breusch–Pagan test (Stata: hettest) ---")
print("LM statistic =", bp_lm)
print("LM p-value   =", bp_lm_pvalue)
print("F statistic  =", bp_f)
print("F p-value    =", bp_f_pvalue)

#if p-value is small (ex: < 0.05), reject H0 -> heteroskedasticity

#robust standard errors (Stata: regress ..., robust)

#HC1 robust SE is the usual match for Stata "robust"
robust_multi = model_multi.get_robustcov_results(cov_type="HC1")

print("\n--- Multiple regression with robust SE (Stata: robust) ---")
print(robust_multi.summary())

#if health_exp is not significant, re-run without it (robust)
#stata: regress life_expectancy fertility gdp_pc, robust

X2 = sm.add_constant(df_multi[["fertility", "gdp_pc"]])
model_multi2 = sm.OLS(y, X2).fit()
robust_multi2 = model_multi2.get_robustcov_results(cov_type="HC1")

print("\n--- Reduced model (drop health_exp) with robust SE ---")
print(robust_multi2.summary())

#3 Non-linear effect of GDP per capita

import numpy as np
from scipy import stats
import statsmodels.api as sm

#keep needed variables and drop missing values (same idea as Stata drop if missing)
df_nl = df[["life_expectancy", "fertility", "gdp_pc"]].dropna().copy()

#create GDP per capita squared 
df_nl["gdp_pc_2"] = df_nl["gdp_pc"] ** 2

#run the regression with the quadratic term
#stata: regress life_expectancy fertility gdp_pc gdp_pc_2, robust
y = df_nl["life_expectancy"]
X = sm.add_constant(df_nl[["fertility", "gdp_pc", "gdp_pc_2"]])

model_nl = sm.OLS(y, X).fit()
robust_nl = model_nl.get_robustcov_results(cov_type="HC1")  

print("\n--- Quadratic model with robust SE (Stata: robust) ---")
print(robust_nl.summary())

#marginal effect evaluated at mean(gdp_pc) (like Stata default margins)
gdp_mean = df_nl["gdp_pc"].mean()

names = robust_nl.model.exog_names              # ["const","fertility","gdp_pc","gdp_pc_2"]
idx_gdp = names.index("gdp_pc")
idx_gdp2 = names.index("gdp_pc_2")

b2 = robust_nl.params[idx_gdp]
b3 = robust_nl.params[idx_gdp2]


marginal_at_mean = b2 + 2 * b3 * gdp_mean

print("\n--- Marginal effect at mean(gdp_pc) (Stata: margins default) ---")
print("mean(gdp_pc) =", gdp_mean)
print("ME = b2 + 2*b3*mean(gdp_pc) =", marginal_at_mean)

#delta-method SE for marginal effect at mean
import pandas as pd

names = robust_nl.model.exog_names
cov = pd.DataFrame(robust_nl.cov_params(), index=names, columns=names)

var_b2 = cov.loc["gdp_pc", "gdp_pc"]
var_b3 = cov.loc["gdp_pc_2", "gdp_pc_2"]
cov_b2b3 = cov.loc["gdp_pc", "gdp_pc_2"]


var_me = var_b2 + (2 * gdp_mean) ** 2 * var_b3 + 2 * (2 * gdp_mean) * cov_b2b3
se_me = np.sqrt(var_me)

#degrees of freedom 
n = int(robust_nl.nobs)
k = int(robust_nl.df_model) + 1
df_t = n - k

t_me = marginal_at_mean / se_me
p_me = 2 * (1 - stats.t.cdf(abs(t_me), df=df_t))

print("\n--- Marginal effect significance (delta method) ---")
print("SE(ME) =", se_me)
print("t-stat =", t_me)
print("p-value =", p_me)
print("df =", df_t)

#manual computation like Stata "display _b[gdp_pc] + 2*_b[gdp_pc_2]*r(mean)"
print("\n--- Manual check (Stata display equivalent) ---")
print(b2 + 2 * b3 * gdp_mean)

#optional interpretation helper:
#effect of +1$ GDP per capita is marginal_at_mean years
#effect of +1000$ GDP per capita is 1000 * marginal_at_mean years
print("\n--- Interpretation at mean(gdp_pc) ---")
print("Effect of +1$ on life expectancy (years) =", marginal_at_mean)
print("Effect of +1000$ on life expectancy (years) =", 1000 * marginal_at_mean)
print("Effect of +1000$ in months ~", 12 * (1000 * marginal_at_mean))

#Statistical inference
print("\n--- Final quadratic model (robust SE) ---")
print(robust_nl.summary())

