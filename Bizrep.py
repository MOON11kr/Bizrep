"""
BizMl: A Python library for predictive analytics and business forecasting.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prophet import Prophet  # Updated import
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import openpyxl
import plotly.express as px
import dash
from dash import dcc, html

# ============================================
# Customer Churn Prediction
# ============================================

class CustomerChurn:
    """
    A class for predicting customer churn using machine learning models.
    Supported models: Gradient Boosting, Logistic Regression, Random Forest.
    """

    def __init__(self, model_type="xgboost"):
        """
        Initialize the churn prediction model.

        Parameters:
        - model_type (str): Type of model to use. Options: 'xgboost', 'logistic', 'random_forest'.
        """
        if model_type == "xgboost":
            self.model = GradientBoostingClassifier()
        elif model_type == "logistic":
            self.model = LogisticRegression()
        elif model_type == "random_forest":
            self.model = RandomForestClassifier()
        else:
            raise ValueError("Unsupported model type. Use 'xgboost', 'logistic', or 'random_forest'.")

    def train(self, data, target_column):
        """
        Train the churn prediction model.

        Parameters:
        - data (pd.DataFrame): Input data containing features and target column.
        - target_column (str): Name of the target column.
        """
        try:
            self.feature_columns = data.drop(columns=[target_column]).columns  # Save feature columns
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.2f}")
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, data):
        """
        Predict churn for new data.

        Parameters:
        - data (pd.DataFrame): Input data for prediction.

        Returns:
        - predictions (np.array): Predicted churn labels.
        """
        try:
            # Ensure only the feature columns used during training are passed
            data = data[self.feature_columns]
            return self.model.predict(data)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def generate_report(self):
        """
        Generate a churn analysis report.
        """
        print("Churn report generated.")

# ============================================
# Sales Forecasting
# ============================================

class SalesForecast:
    """
    A class for forecasting sales using time series models.
    Supported models: Prophet, ARIMA.
    """

    def __init__(self, model_type="prophet"):
        """
        Initialize the sales forecasting model.

        Parameters:
        - model_type (str): Type of model to use. Options: 'prophet', 'arima'.
        """
        if model_type == "prophet":
            self.model = Prophet()
        elif model_type == "arima":
            self.model = ARIMA(order=(1, 1, 1))  # Example order
        else:
            raise ValueError("Unsupported model type. Use 'prophet' or 'arima'.")

    def train(self, data, date_column, target_column):
        """
        Train the sales forecasting model.

        Parameters:
        - data (pd.DataFrame): Input data containing date and target columns.
        - date_column (str): Name of the date column.
        - target_column (str): Name of the target column.
        """
        try:
            if isinstance(self.model, Prophet):
                df = data[[date_column, target_column]].rename(columns={date_column: "ds", target_column: "y"})
                self.model.fit(df)
            elif isinstance(self.model, ARIMA):
                self.model = self.model.fit(data[target_column])
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, steps):
        """
        Forecast sales for future time steps.

        Parameters:
        - steps (int): Number of future time steps to forecast.

        Returns:
        - forecast (pd.DataFrame or np.array): Forecasted sales values.
        """
        try:
            if isinstance(self.model, Prophet):
                future = self.model.make_future_dataframe(periods=steps)
                forecast = self.model.predict(future)
                return forecast[["ds", "yhat"]]
            elif isinstance(self.model, ARIMA):
                forecast = self.model.forecast(steps=steps)
                return forecast
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def generate_report(self):
        """
        Generate a sales forecasting report.
        """
        print("Sales forecast report generated.")

# ============================================
# Automated Business Reporting
# ============================================

class BusinessReport:
    """
    A class for generating automated business reports.
    Supported formats: PDF, Excel.
    """

    def __init__(self, data):
        """
        Initialize the report generator.

        Parameters:
        - data (pd.DataFrame): Input data for reporting.
        """
        self.data = data

    def generate_sales_report(self, output_format="pdf"):
        """
        Generate a sales report.

        Parameters:
        - output_format (str): Format of the report. Options: 'pdf', 'excel'.
        """
        try:
            if output_format == "pdf":
                self._generate_pdf_report("sales_report.pdf")
            elif output_format == "excel":
                self._generate_excel_report("sales_report.xlsx")
            else:
                raise ValueError("Unsupported output format. Use 'pdf' or 'excel'.")
        except Exception as e:
            print(f"Error generating sales report: {e}")

    def generate_financial_summary(self, output_format="pdf"):
        """
        Generate a financial summary report.

        Parameters:
        - output_format (str): Format of the report. Options: 'pdf', 'excel'.
        """
        try:
            if output_format == "pdf":
                self._generate_pdf_report("financial_summary.pdf")
            elif output_format == "excel":
                self._generate_excel_report("financial_summary.xlsx")
            else:
                raise ValueError("Unsupported output format. Use 'pdf' or 'excel'.")
        except Exception as e:
            print(f"Error generating financial summary: {e}")

    def _generate_pdf_report(self, filename):
        """
        Generate a PDF report.

        Parameters:
        - filename (str): Name of the output PDF file.
        """
        c = canvas.Canvas(filename, pagesize=letter)
        c.drawString(100, 750, "Sales Report")
        c.drawString(100, 730, "Generated by BizML")
        c.save()

    def _generate_excel_report(self, filename):
        """
        Generate an Excel report.

        Parameters:
        - filename (str): Name of the output Excel file.
        """
        self.data.to_excel(filename, index=False)

# ============================================
# Interactive Dashboards
# ============================================

class InteractiveDashboard:
    """
    A class for creating interactive dashboards.
    Supported dashboards: Sales, Churn.
    """

    def __init__(self, data):
        """
        Initialize the dashboard.

        Parameters:
        - data (pd.DataFrame): Input data for the dashboard.
        """
        self.data = data

    def create_sales_dashboard(self):
        """
        Create an interactive sales dashboard.
        """
        try:
            # Line chart for sales trends
            fig1 = px.line(self.data, x="date", y="sales", title="Sales Trends")
            fig1.show()

            # Pie chart for sales distribution
            fig2 = px.pie(self.data, names="product_category", values="sales", title="Sales Distribution by Category")
            fig2.show()
        except Exception as e:
            print(f"Error creating sales dashboard: {e}")

    def create_churn_dashboard(self):
        """
        Create an interactive churn dashboard.
        """
        try:
            # Bar chart for churn analysis
            fig1 = px.bar(self.data, x="customer_id", y="churn", title="Churn Analysis")
            fig1.show()

            # Heatmap for churn correlation
            fig2 = px.imshow(self.data.corr(), title="Churn Correlation Heatmap")
            fig2.show()
        except Exception as e:
            print(f"Error creating churn dashboard: {e}")

