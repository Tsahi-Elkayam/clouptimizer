"""
Trend Analysis and Cost Forecasting
Provides historical trend analysis and predictive cost forecasting.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Trend direction types"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class ForecastConfidence(str, Enum):
    """Forecast confidence levels"""
    HIGH = "high"  # <10% error margin
    MEDIUM = "medium"  # 10-20% error margin
    LOW = "low"  # >20% error margin


class SeasonalPattern(str, Enum):
    """Seasonal pattern types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NONE = "none"


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    resource_id: str
    metric_name: str
    period_start: date
    period_end: date
    trend_direction: TrendDirection
    trend_strength: float  # 0-1, strength of trend
    average_value: float
    min_value: float
    max_value: float
    std_deviation: float
    growth_rate: float  # Percentage per period
    seasonality: SeasonalPattern
    anomalies: List[Dict[str, Any]]
    change_points: List[date]
    
    @property
    def is_growing(self) -> bool:
        return self.trend_direction == TrendDirection.INCREASING
    
    @property
    def is_seasonal(self) -> bool:
        return self.seasonality != SeasonalPattern.NONE


@dataclass
class CostForecast:
    """Cost forecast results"""
    resource_id: str
    forecast_date: date
    forecast_horizon_days: int
    current_cost: float
    forecasted_costs: List[float]  # Daily forecasts
    confidence_intervals: List[Tuple[float, float]]  # Lower and upper bounds
    confidence_level: ForecastConfidence
    forecast_method: str
    accuracy_metrics: Dict[str, float]
    influencing_factors: List[str]
    recommendations: List[str]
    
    @property
    def forecasted_monthly_cost(self) -> float:
        """Get forecasted cost for next 30 days"""
        return sum(self.forecasted_costs[:30])
    
    @property
    def forecasted_annual_cost(self) -> float:
        """Get forecasted annual cost"""
        daily_avg = sum(self.forecasted_costs) / len(self.forecasted_costs)
        return daily_avg * 365
    
    @property
    def cost_increase_percentage(self) -> float:
        """Calculate percentage increase from current to forecasted"""
        if self.current_cost > 0:
            return ((self.forecasted_monthly_cost - self.current_cost) / self.current_cost) * 100
        return 0


@dataclass
class BudgetAlert:
    """Budget alert based on forecasting"""
    alert_id: str
    alert_type: str  # budget_exceed, unusual_spike, trend_alert
    severity: str  # critical, high, medium, low
    resource_id: str
    current_spend: float
    forecasted_spend: float
    budget_threshold: float
    days_until_exceed: Optional[int]
    description: str
    recommended_actions: List[str]
    alert_date: datetime = field(default_factory=datetime.now)
    
    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"


class TrendForecastingEngine:
    """
    Advanced trend analysis and cost forecasting engine.
    Uses multiple forecasting methods for accurate predictions.
    """
    
    def __init__(self):
        """Initialize forecasting engine"""
        self.linear_model = LinearRegression()
        self.polynomial_model = None
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.prophet_model = None
        self.trend_analyses = []
        self.forecasts = []
        self.budget_alerts = []
    
    def analyze_trends(self, historical_data: List[Dict], 
                      metric: str = 'cost') -> List[TrendAnalysis]:
        """
        Analyze historical trends in data.
        
        Args:
            historical_data: Historical metrics data
            metric: Metric to analyze
            
        Returns:
            List of trend analyses
        """
        self.trend_analyses = []
        
        # Group data by resource
        grouped_data = self._group_by_resource(historical_data)
        
        for resource_id, data in grouped_data.items():
            if len(data) < 7:  # Need at least a week of data
                continue
            
            analysis = self._analyze_resource_trend(resource_id, data, metric)
            if analysis:
                self.trend_analyses.append(analysis)
        
        return self.trend_analyses
    
    def _analyze_resource_trend(self, resource_id: str, data: List[Dict], 
                               metric: str) -> Optional[TrendAnalysis]:
        """Analyze trend for a single resource"""
        try:
            # Convert to pandas DataFrame for easier analysis
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            if metric not in df.columns:
                return None
            
            # Basic statistics
            values = df[metric].values
            dates = df['date'].values
            
            # Trend analysis using linear regression
            X = np.arange(len(values)).reshape(-1, 1)
            y = values
            
            self.linear_model.fit(X, y)
            trend_slope = self.linear_model.coef_[0]
            
            # Determine trend direction and strength
            trend_direction = self._determine_trend_direction(trend_slope, values)
            trend_strength = self._calculate_trend_strength(values, self.linear_model.predict(X))
            
            # Calculate growth rate
            if values[0] != 0:
                growth_rate = ((values[-1] - values[0]) / values[0]) * 100
            else:
                growth_rate = 0
            
            # Detect seasonality
            seasonality = self._detect_seasonality(df, metric)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(df, metric)
            
            # Detect change points
            change_points = self._detect_change_points(values, dates)
            
            return TrendAnalysis(
                resource_id=resource_id,
                metric_name=metric,
                period_start=df['date'].min().date(),
                period_end=df['date'].max().date(),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                average_value=np.mean(values),
                min_value=np.min(values),
                max_value=np.max(values),
                std_deviation=np.std(values),
                growth_rate=growth_rate,
                seasonality=seasonality,
                anomalies=anomalies,
                change_points=change_points
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {resource_id}: {e}")
            return None
    
    def forecast_costs(self, historical_data: List[Dict], 
                      horizon_days: int = 30,
                      method: str = 'auto') -> List[CostForecast]:
        """
        Forecast future costs based on historical data.
        
        Args:
            historical_data: Historical cost data
            horizon_days: Number of days to forecast
            method: Forecasting method ('linear', 'polynomial', 'prophet', 'ensemble', 'auto')
            
        Returns:
            List of cost forecasts
        """
        self.forecasts = []
        
        # Group data by resource
        grouped_data = self._group_by_resource(historical_data)
        
        for resource_id, data in grouped_data.items():
            if len(data) < 14:  # Need at least 2 weeks of data
                continue
            
            forecast = self._forecast_resource_cost(resource_id, data, horizon_days, method)
            if forecast:
                self.forecasts.append(forecast)
        
        return self.forecasts
    
    def _forecast_resource_cost(self, resource_id: str, data: List[Dict],
                               horizon_days: int, method: str) -> Optional[CostForecast]:
        """Forecast cost for a single resource"""
        try:
            # Prepare data
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            if 'cost' not in df.columns:
                return None
            
            current_cost = df['cost'].iloc[-30:].mean() if len(df) >= 30 else df['cost'].mean()
            
            # Select forecasting method
            if method == 'auto':
                method = self._select_best_method(df)
            
            # Generate forecast
            if method == 'prophet':
                forecasts, confidence_intervals = self._prophet_forecast(df, horizon_days)
            elif method == 'polynomial':
                forecasts, confidence_intervals = self._polynomial_forecast(df, horizon_days)
            elif method == 'ensemble':
                forecasts, confidence_intervals = self._ensemble_forecast(df, horizon_days)
            else:  # linear
                forecasts, confidence_intervals = self._linear_forecast(df, horizon_days)
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(df, method)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(accuracy_metrics)
            
            # Identify influencing factors
            influencing_factors = self._identify_influencing_factors(df)
            
            # Generate recommendations
            recommendations = self._generate_forecast_recommendations(
                current_cost, forecasts, confidence_level
            )
            
            return CostForecast(
                resource_id=resource_id,
                forecast_date=date.today(),
                forecast_horizon_days=horizon_days,
                current_cost=current_cost * 30,  # Monthly cost
                forecasted_costs=forecasts,
                confidence_intervals=confidence_intervals,
                confidence_level=confidence_level,
                forecast_method=method,
                accuracy_metrics=accuracy_metrics,
                influencing_factors=influencing_factors,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error forecasting cost for {resource_id}: {e}")
            return None
    
    def _linear_forecast(self, df: pd.DataFrame, horizon_days: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Linear regression forecast"""
        # Prepare data
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['cost'].values
        
        # Fit model
        self.linear_model.fit(X, y)
        
        # Generate predictions
        future_X = np.arange(len(df), len(df) + horizon_days).reshape(-1, 1)
        predictions = self.linear_model.predict(future_X)
        
        # Calculate confidence intervals (simplified)
        std_error = np.std(y - self.linear_model.predict(X))
        confidence_intervals = [
            (pred - 1.96 * std_error, pred + 1.96 * std_error)
            for pred in predictions
        ]
        
        return predictions.tolist(), confidence_intervals
    
    def _polynomial_forecast(self, df: pd.DataFrame, horizon_days: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Polynomial regression forecast"""
        # Prepare data
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['cost'].values
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate predictions
        future_X = np.arange(len(df), len(df) + horizon_days).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        predictions = model.predict(future_X_poly)
        
        # Calculate confidence intervals
        std_error = np.std(y - model.predict(X_poly))
        confidence_intervals = [
            (pred - 1.96 * std_error, pred + 1.96 * std_error)
            for pred in predictions
        ]
        
        return predictions.tolist(), confidence_intervals
    
    def _prophet_forecast(self, df: pd.DataFrame, horizon_days: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Facebook Prophet forecast"""
        try:
            # Prepare data for Prophet
            prophet_df = df[['date', 'cost']].rename(columns={'date': 'ds', 'cost': 'y'})
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95
            )
            
            model.fit(prophet_df)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=horizon_days)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract predictions for future dates
            future_forecast = forecast.iloc[-horizon_days:]
            predictions = future_forecast['yhat'].tolist()
            confidence_intervals = list(zip(
                future_forecast['yhat_lower'].tolist(),
                future_forecast['yhat_upper'].tolist()
            ))
            
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            # Fallback to linear
            return self._linear_forecast(df, horizon_days)
    
    def _ensemble_forecast(self, df: pd.DataFrame, horizon_days: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Ensemble forecast combining multiple methods"""
        # Get forecasts from different methods
        linear_pred, linear_ci = self._linear_forecast(df, horizon_days)
        poly_pred, poly_ci = self._polynomial_forecast(df, horizon_days)
        prophet_pred, prophet_ci = self._prophet_forecast(df, horizon_days)
        
        # Combine predictions (weighted average)
        weights = [0.3, 0.3, 0.4]  # Linear, Polynomial, Prophet
        ensemble_predictions = []
        ensemble_ci = []
        
        for i in range(horizon_days):
            pred = (weights[0] * linear_pred[i] + 
                   weights[1] * poly_pred[i] + 
                   weights[2] * prophet_pred[i])
            ensemble_predictions.append(pred)
            
            # Combine confidence intervals
            lower = min(linear_ci[i][0], poly_ci[i][0], prophet_ci[i][0])
            upper = max(linear_ci[i][1], poly_ci[i][1], prophet_ci[i][1])
            ensemble_ci.append((lower, upper))
        
        return ensemble_predictions, ensemble_ci
    
    def _select_best_method(self, df: pd.DataFrame) -> str:
        """Select best forecasting method based on data characteristics"""
        # Check data length
        if len(df) < 30:
            return 'linear'
        elif len(df) < 90:
            return 'polynomial'
        else:
            # Check for seasonality
            if self._has_strong_seasonality(df):
                return 'prophet'
            else:
                return 'ensemble'
    
    def _has_strong_seasonality(self, df: pd.DataFrame) -> bool:
        """Check if data has strong seasonality"""
        try:
            if len(df) < 60:  # Need enough data for seasonal decomposition
                return False
            
            # Perform seasonal decomposition
            result = seasonal_decompose(df['cost'], model='additive', period=7)
            
            # Check if seasonal component is significant
            seasonal_strength = np.std(result.seasonal) / np.std(df['cost'])
            return seasonal_strength > 0.3
            
        except:
            return False
    
    def _determine_trend_direction(self, slope: float, values: np.ndarray) -> TrendDirection:
        """Determine trend direction from slope and values"""
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        if cv > 0.5:  # High coefficient of variation
            return TrendDirection.VOLATILE
        elif abs(slope) < 0.01 * np.mean(values):  # Less than 1% change
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING
    
    def _calculate_trend_strength(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate strength of trend (R-squared)"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0, min(1, r_squared))
    
    def _detect_seasonality(self, df: pd.DataFrame, metric: str) -> SeasonalPattern:
        """Detect seasonal patterns in data"""
        try:
            if len(df) < 14:
                return SeasonalPattern.NONE
            
            # Check for weekly pattern
            if len(df) >= 14:
                weekly_avg = df.groupby(df['date'].dt.dayofweek)[metric].mean()
                weekly_cv = weekly_avg.std() / weekly_avg.mean() if weekly_avg.mean() != 0 else 0
                
                if weekly_cv > 0.2:
                    return SeasonalPattern.WEEKLY
            
            # Check for monthly pattern
            if len(df) >= 60:
                monthly_avg = df.groupby(df['date'].dt.day)[metric].mean()
                monthly_cv = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0
                
                if monthly_cv > 0.2:
                    return SeasonalPattern.MONTHLY
            
            return SeasonalPattern.NONE
            
        except:
            return SeasonalPattern.NONE
    
    def _detect_anomalies(self, df: pd.DataFrame, metric: str) -> List[Dict]:
        """Detect anomalies in time series data"""
        anomalies = []
        
        try:
            values = df[metric].values
            mean = np.mean(values)
            std = np.std(values)
            
            # Use 3-sigma rule for anomaly detection
            for i, row in df.iterrows():
                value = row[metric]
                if abs(value - mean) > 3 * std:
                    anomalies.append({
                        'date': row['date'].isoformat(),
                        'value': value,
                        'expected_range': (mean - 2*std, mean + 2*std),
                        'deviation': abs(value - mean) / std
                    })
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def _detect_change_points(self, values: np.ndarray, dates: np.ndarray) -> List[date]:
        """Detect significant change points in trend"""
        change_points = []
        
        try:
            if len(values) < 20:
                return change_points
            
            # Use simple difference method
            window = 7
            for i in range(window, len(values) - window):
                before_mean = np.mean(values[i-window:i])
                after_mean = np.mean(values[i:i+window])
                
                # Check for significant change (>20%)
                if before_mean != 0:
                    change = abs(after_mean - before_mean) / before_mean
                    if change > 0.2:
                        change_points.append(pd.Timestamp(dates[i]).date())
            
        except Exception as e:
            logger.error(f"Error detecting change points: {e}")
        
        return change_points
    
    def _calculate_accuracy_metrics(self, df: pd.DataFrame, method: str) -> Dict[str, float]:
        """Calculate forecast accuracy metrics using backtesting"""
        metrics = {
            'mape': 0,  # Mean Absolute Percentage Error
            'rmse': 0,  # Root Mean Square Error
            'mae': 0,   # Mean Absolute Error
        }
        
        try:
            if len(df) < 30:
                return metrics
            
            # Split data for backtesting
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Generate predictions for test period
            if method == 'linear':
                predictions, _ = self._linear_forecast(train_df, len(test_df))
            else:
                predictions, _ = self._linear_forecast(train_df, len(test_df))
            
            # Calculate metrics
            actual = test_df['cost'].values
            predicted = np.array(predictions[:len(actual)])
            
            metrics['mae'] = np.mean(np.abs(actual - predicted))
            metrics['rmse'] = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # MAPE (avoiding division by zero)
            non_zero_actual = actual[actual != 0]
            non_zero_predicted = predicted[actual != 0]
            if len(non_zero_actual) > 0:
                metrics['mape'] = np.mean(np.abs((non_zero_actual - non_zero_predicted) / non_zero_actual)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
        
        return metrics
    
    def _determine_confidence_level(self, accuracy_metrics: Dict[str, float]) -> ForecastConfidence:
        """Determine forecast confidence level based on accuracy"""
        mape = accuracy_metrics.get('mape', 100)
        
        if mape < 10:
            return ForecastConfidence.HIGH
        elif mape < 20:
            return ForecastConfidence.MEDIUM
        else:
            return ForecastConfidence.LOW
    
    def _identify_influencing_factors(self, df: pd.DataFrame) -> List[str]:
        """Identify factors influencing cost trends"""
        factors = []
        
        # Check for day-of-week effect
        if len(df) >= 14:
            weekday_avg = df[df['date'].dt.dayofweek < 5]['cost'].mean()
            weekend_avg = df[df['date'].dt.dayofweek >= 5]['cost'].mean()
            
            if abs(weekday_avg - weekend_avg) / weekday_avg > 0.2:
                factors.append("Weekday vs weekend usage patterns")
        
        # Check for growth trend
        if len(df) >= 30:
            first_week = df.iloc[:7]['cost'].mean()
            last_week = df.iloc[-7:]['cost'].mean()
            
            if last_week > first_week * 1.1:
                factors.append("Consistent growth trend")
            elif last_week < first_week * 0.9:
                factors.append("Declining usage trend")
        
        # Check for volatility
        cv = df['cost'].std() / df['cost'].mean() if df['cost'].mean() != 0 else 0
        if cv > 0.5:
            factors.append("High cost volatility")
        
        return factors
    
    def _generate_forecast_recommendations(self, current_cost: float,
                                          forecasts: List[float],
                                          confidence: ForecastConfidence) -> List[str]:
        """Generate recommendations based on forecast"""
        recommendations = []
        
        forecasted_monthly = sum(forecasts[:30]) if len(forecasts) >= 30 else sum(forecasts)
        
        # Check for cost increase
        if forecasted_monthly > current_cost * 30 * 1.2:
            recommendations.append("Significant cost increase predicted - review resource usage")
            recommendations.append("Consider implementing cost controls or budgets")
            
            if confidence == ForecastConfidence.HIGH:
                recommendations.append("High confidence forecast - take preventive action")
        
        # Check for optimization opportunity
        elif forecasted_monthly > current_cost * 30 * 1.05:
            recommendations.append("Moderate cost increase expected")
            recommendations.append("Review optimization recommendations")
        
        # Check for cost decrease
        elif forecasted_monthly < current_cost * 30 * 0.8:
            recommendations.append("Cost decrease predicted - verify resource requirements")
            recommendations.append("Consider reallocating saved budget")
        
        return recommendations
    
    def detect_budget_alerts(self, forecasts: List[CostForecast],
                            budgets: Dict[str, float]) -> List[BudgetAlert]:
        """
        Detect budget alerts based on forecasts.
        
        Args:
            forecasts: List of cost forecasts
            budgets: Budget thresholds by resource or total
            
        Returns:
            List of budget alerts
        """
        self.budget_alerts = []
        alert_counter = 0
        
        for forecast in forecasts:
            # Check if resource has a budget
            budget = budgets.get(forecast.resource_id, budgets.get('total', 0))
            
            if budget <= 0:
                continue
            
            # Check current spending
            if forecast.current_cost > budget * 0.8:
                alert_counter += 1
                self.budget_alerts.append(BudgetAlert(
                    alert_id=f"alert-{alert_counter:04d}",
                    alert_type="budget_exceed",
                    severity="high" if forecast.current_cost > budget else "medium",
                    resource_id=forecast.resource_id,
                    current_spend=forecast.current_cost,
                    forecasted_spend=forecast.forecasted_monthly_cost,
                    budget_threshold=budget,
                    days_until_exceed=self._calculate_days_until_exceed(forecast, budget),
                    description=f"Resource approaching budget limit",
                    recommended_actions=[
                        "Review resource utilization",
                        "Implement cost optimization recommendations",
                        "Consider adjusting budget or resource allocation"
                    ]
                ))
            
            # Check forecasted spending
            if forecast.forecasted_monthly_cost > budget:
                alert_counter += 1
                self.budget_alerts.append(BudgetAlert(
                    alert_id=f"alert-{alert_counter:04d}",
                    alert_type="forecast_exceed",
                    severity="critical" if forecast.confidence_level == ForecastConfidence.HIGH else "high",
                    resource_id=forecast.resource_id,
                    current_spend=forecast.current_cost,
                    forecasted_spend=forecast.forecasted_monthly_cost,
                    budget_threshold=budget,
                    days_until_exceed=self._calculate_days_until_exceed(forecast, budget),
                    description=f"Forecast predicts budget will be exceeded",
                    recommended_actions=[
                        "Take immediate action to reduce costs",
                        "Review and adjust resource configurations",
                        "Consider Reserved Instances or Savings Plans"
                    ]
                ))
        
        return self.budget_alerts
    
    def _calculate_days_until_exceed(self, forecast: CostForecast, budget: float) -> Optional[int]:
        """Calculate days until budget is exceeded"""
        cumulative_cost = 0
        daily_budget = budget / 30
        
        for i, daily_cost in enumerate(forecast.forecasted_costs):
            cumulative_cost += daily_cost
            if cumulative_cost > budget:
                return i + 1
        
        return None
    
    def _group_by_resource(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group data by resource ID"""
        grouped = defaultdict(list)
        
        for item in data:
            resource_id = item.get('resource_id', 'unknown')
            grouped[resource_id].append(item)
        
        return dict(grouped)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of trends and forecasts"""
        return {
            'trend_analysis': {
                'total_resources_analyzed': len(self.trend_analyses),
                'increasing_trends': sum(1 for t in self.trend_analyses if t.is_growing),
                'seasonal_patterns': sum(1 for t in self.trend_analyses if t.is_seasonal),
                'volatile_resources': sum(1 for t in self.trend_analyses 
                                        if t.trend_direction == TrendDirection.VOLATILE),
                'average_growth_rate': np.mean([t.growth_rate for t in self.trend_analyses])
                                      if self.trend_analyses else 0
            },
            'forecasts': {
                'total_forecasts': len(self.forecasts),
                'high_confidence': sum(1 for f in self.forecasts 
                                     if f.confidence_level == ForecastConfidence.HIGH),
                'total_current_cost': sum(f.current_cost for f in self.forecasts),
                'total_forecasted_cost': sum(f.forecasted_monthly_cost for f in self.forecasts),
                'expected_increase': sum(f.forecasted_monthly_cost - f.current_cost 
                                       for f in self.forecasts)
            },
            'alerts': {
                'total_alerts': len(self.budget_alerts),
                'critical_alerts': sum(1 for a in self.budget_alerts if a.is_critical),
                'resources_at_risk': len(set(a.resource_id for a in self.budget_alerts))
            }
        }