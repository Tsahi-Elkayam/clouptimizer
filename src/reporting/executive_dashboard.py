"""
Executive dashboard and advanced reporting.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from jinja2 import Template
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import boto3

logger = logging.getLogger(__name__)


@dataclass
class ReportMetrics:
    """Metrics for executive reporting."""
    total_spend: float
    projected_spend: float
    realized_savings: float
    potential_savings: float
    optimization_score: float
    compliance_score: float
    resource_utilization: float
    waste_percentage: float
    mtd_spend: float  # Month-to-date
    ytd_spend: float  # Year-to-date
    forecast_accuracy: float
    cost_by_service: Dict[str, float]
    cost_by_team: Dict[str, float]
    cost_by_environment: Dict[str, float]
    top_recommendations: List[Dict[str, Any]]
    trend_data: List[Dict[str, Any]]


class ExecutiveDashboard:
    """Generate executive-level dashboards and reports."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = boto3.Session()
        
        # Report templates
        self.templates = {
            'executive_summary': self._load_template('executive_summary.html'),
            'cost_analysis': self._load_template('cost_analysis.html'),
            'optimization_report': self._load_template('optimization_report.html'),
            'compliance_report': self._load_template('compliance_report.html')
        }
    
    def _load_template(self, template_name: str) -> Optional[Template]:
        """Load report template."""
        # In production, load from template files
        # For now, return inline templates
        if template_name == 'executive_summary.html':
            return Template("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Executive Cost Summary</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background: #2c3e50; color: white; padding: 20px; }
                    .metric-card { 
                        display: inline-block; 
                        background: #f8f9fa; 
                        padding: 20px; 
                        margin: 10px;
                        border-radius: 8px;
                        min-width: 200px;
                    }
                    .metric-value { font-size: 32px; font-weight: bold; }
                    .metric-label { color: #6c757d; margin-top: 5px; }
                    .chart-container { margin: 20px 0; }
                    .recommendation { 
                        background: #fff3cd; 
                        padding: 15px; 
                        margin: 10px 0;
                        border-left: 4px solid #ffc107;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Cloud Cost Executive Summary</h1>
                    <p>Report Date: {{ report_date }}</p>
                </div>
                
                <h2>Key Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">${{ "{:,.0f}".format(metrics.total_spend) }}</div>
                        <div class="metric-label">Current Month Spend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{ "{:,.0f}".format(metrics.projected_spend) }}</div>
                        <div class="metric-label">Projected Month End</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{ "{:,.0f}".format(metrics.realized_savings) }}</div>
                        <div class="metric-label">Realized Savings</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "{:.1f}%".format(metrics.optimization_score) }}</div>
                        <div class="metric-label">Optimization Score</div>
                    </div>
                </div>
                
                <h2>Cost Breakdown</h2>
                <div class="chart-container">
                    {{ cost_breakdown_chart | safe }}
                </div>
                
                <h2>Top Optimization Opportunities</h2>
                {% for rec in metrics.top_recommendations[:5] %}
                <div class="recommendation">
                    <strong>{{ rec.title }}</strong><br>
                    {{ rec.description }}<br>
                    Potential Savings: ${{ "{:,.0f}".format(rec.savings) }}/month
                </div>
                {% endfor %}
                
                <h2>Trend Analysis</h2>
                <div class="chart-container">
                    {{ trend_chart | safe }}
                </div>
            </body>
            </html>
            """)
        return None
    
    def generate_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary report."""
        # Calculate metrics
        metrics = self._calculate_metrics(data)
        
        # Generate charts
        charts = {
            'cost_breakdown_chart': self._generate_cost_breakdown_chart(metrics),
            'trend_chart': self._generate_trend_chart(metrics),
            'utilization_chart': self._generate_utilization_chart(metrics),
            'savings_chart': self._generate_savings_chart(metrics)
        }
        
        # Generate insights
        insights = self._generate_executive_insights(metrics)
        
        # Create dashboard data
        dashboard = {
            'metrics': metrics,
            'charts': charts,
            'insights': insights,
            'generated_at': datetime.utcnow().isoformat(),
            'report_period': self._get_report_period()
        }
        
        return dashboard
    
    def _calculate_metrics(self, data: Dict[str, Any]) -> ReportMetrics:
        """Calculate executive metrics."""
        # Extract cost data
        costs = data.get('costs', {})
        resources = data.get('resources', [])
        recommendations = data.get('recommendations', [])
        compliance = data.get('compliance', {})
        
        # Calculate spend metrics
        total_spend = sum(costs.get('daily_costs', []))
        mtd_spend = sum(costs.get('mtd_costs', []))
        ytd_spend = sum(costs.get('ytd_costs', []))
        
        # Project month-end spend
        days_in_month = 30
        current_day = datetime.utcnow().day
        projected_spend = (mtd_spend / current_day) * days_in_month if current_day > 0 else 0
        
        # Calculate savings
        realized_savings = sum(r.get('realized_savings', 0) for r in recommendations if r.get('implemented'))
        potential_savings = sum(r.get('estimated_savings', 0) for r in recommendations if not r.get('implemented'))
        
        # Calculate scores
        optimization_score = self._calculate_optimization_score(resources, recommendations)
        compliance_score = compliance.get('overall_score', 0)
        
        # Calculate utilization
        resource_utilization = self._calculate_resource_utilization(resources)
        
        # Calculate waste
        waste_amount = sum(r.get('waste_amount', 0) for r in resources if r.get('is_waste'))
        waste_percentage = (waste_amount / total_spend * 100) if total_spend > 0 else 0
        
        # Cost breakdown
        cost_by_service = self._aggregate_costs_by_dimension(costs, 'service')
        cost_by_team = self._aggregate_costs_by_dimension(costs, 'team')
        cost_by_environment = self._aggregate_costs_by_dimension(costs, 'environment')
        
        # Top recommendations
        top_recommendations = sorted(
            recommendations,
            key=lambda x: x.get('estimated_savings', 0),
            reverse=True
        )[:10]
        
        # Trend data
        trend_data = self._prepare_trend_data(costs)
        
        # Forecast accuracy
        forecast_accuracy = self._calculate_forecast_accuracy(costs)
        
        return ReportMetrics(
            total_spend=total_spend,
            projected_spend=projected_spend,
            realized_savings=realized_savings,
            potential_savings=potential_savings,
            optimization_score=optimization_score,
            compliance_score=compliance_score,
            resource_utilization=resource_utilization,
            waste_percentage=waste_percentage,
            mtd_spend=mtd_spend,
            ytd_spend=ytd_spend,
            forecast_accuracy=forecast_accuracy,
            cost_by_service=cost_by_service,
            cost_by_team=cost_by_team,
            cost_by_environment=cost_by_environment,
            top_recommendations=top_recommendations,
            trend_data=trend_data
        )
    
    def _calculate_optimization_score(self, resources: List[Dict], 
                                     recommendations: List[Dict]) -> float:
        """Calculate overall optimization score."""
        if not resources:
            return 0
        
        # Factors for optimization score
        utilization_score = np.mean([r.get('utilization', 0) for r in resources])
        
        # Right-sizing score
        right_sized = sum(1 for r in resources if r.get('is_right_sized'))
        right_sizing_score = (right_sized / len(resources)) * 100
        
        # Implementation score
        total_recommendations = len(recommendations)
        implemented = sum(1 for r in recommendations if r.get('implemented'))
        implementation_score = (implemented / total_recommendations * 100) if total_recommendations > 0 else 100
        
        # Waste reduction score
        waste_count = sum(1 for r in resources if r.get('is_waste'))
        waste_score = ((len(resources) - waste_count) / len(resources)) * 100
        
        # Weighted average
        weights = {
            'utilization': 0.3,
            'right_sizing': 0.25,
            'implementation': 0.25,
            'waste': 0.2
        }
        
        score = (
            utilization_score * weights['utilization'] +
            right_sizing_score * weights['right_sizing'] +
            implementation_score * weights['implementation'] +
            waste_score * weights['waste']
        )
        
        return min(100, max(0, score))
    
    def _calculate_resource_utilization(self, resources: List[Dict]) -> float:
        """Calculate average resource utilization."""
        if not resources:
            return 0
        
        utilizations = [r.get('utilization', 0) for r in resources if 'utilization' in r]
        return np.mean(utilizations) if utilizations else 0
    
    def _aggregate_costs_by_dimension(self, costs: Dict, dimension: str) -> Dict[str, float]:
        """Aggregate costs by dimension."""
        aggregated = {}
        
        for item in costs.get('items', []):
            key = item.get(dimension, 'Unknown')
            value = item.get('cost', 0)
            aggregated[key] = aggregated.get(key, 0) + value
        
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    
    def _prepare_trend_data(self, costs: Dict) -> List[Dict[str, Any]]:
        """Prepare trend data for visualization."""
        trend_data = []
        
        for i in range(30):  # Last 30 days
            date = datetime.utcnow() - timedelta(days=i)
            daily_cost = costs.get('daily_costs', {}).get(date.strftime('%Y-%m-%d'), 0)
            
            trend_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'cost': daily_cost,
                'day_of_week': date.strftime('%A'),
                'is_weekend': date.weekday() >= 5
            })
        
        return sorted(trend_data, key=lambda x: x['date'])
    
    def _calculate_forecast_accuracy(self, costs: Dict) -> float:
        """Calculate forecast accuracy."""
        forecasts = costs.get('forecasts', [])
        actuals = costs.get('actuals', [])
        
        if not forecasts or not actuals:
            return 0
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        errors = []
        for forecast, actual in zip(forecasts[-30:], actuals[-30:]):
            if actual > 0:
                error = abs(forecast - actual) / actual
                errors.append(error)
        
        if errors:
            mape = np.mean(errors) * 100
            accuracy = max(0, 100 - mape)
            return accuracy
        
        return 0
    
    def _generate_cost_breakdown_chart(self, metrics: ReportMetrics) -> str:
        """Generate cost breakdown pie chart."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cost by Service', 'Cost by Environment'),
            specs=[[{'type': 'pie'}, {'type': 'pie'}]]
        )
        
        # Service breakdown
        services = list(metrics.cost_by_service.keys())[:10]
        service_values = [metrics.cost_by_service[s] for s in services]
        
        fig.add_trace(
            go.Pie(
                labels=services,
                values=service_values,
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            ),
            row=1, col=1
        )
        
        # Environment breakdown
        environments = list(metrics.cost_by_environment.keys())
        env_values = [metrics.cost_by_environment[e] for e in environments]
        
        fig.add_trace(
            go.Pie(
                labels=environments,
                values=env_values,
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Pastel)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Cost Distribution Analysis",
            showlegend=True,
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="cost-breakdown")
    
    def _generate_trend_chart(self, metrics: ReportMetrics) -> str:
        """Generate cost trend chart."""
        df = pd.DataFrame(metrics.trend_data)
        
        fig = go.Figure()
        
        # Actual costs
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cost'],
            mode='lines+markers',
            name='Daily Cost',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6)
        ))
        
        # Add moving average
        df['ma7'] = df['cost'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['ma7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#A23B72', width=2, dash='dash')
        ))
        
        # Add forecast
        last_date = pd.to_datetime(df['date'].iloc[-1])
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)
        forecast_values = [df['cost'].mean()] * 7  # Simple forecast
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#F18F01', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title="Cost Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            hovermode='x unified',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="trend-chart")
    
    def _generate_utilization_chart(self, metrics: ReportMetrics) -> str:
        """Generate resource utilization chart."""
        categories = ['CPU', 'Memory', 'Storage', 'Network', 'Database']
        utilization = [75, 82, 45, 60, 70]  # Example data
        optimal = [80, 80, 80, 80, 80]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Utilization',
            x=categories,
            y=utilization,
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Scatter(
            name='Optimal Level',
            x=categories,
            y=optimal,
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Resource Utilization Analysis",
            xaxis_title="Resource Type",
            yaxis_title="Utilization (%)",
            yaxis_range=[0, 100],
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="utilization-chart")
    
    def _generate_savings_chart(self, metrics: ReportMetrics) -> str:
        """Generate savings opportunity chart."""
        recommendations = metrics.top_recommendations[:5]
        
        names = [r.get('title', 'Unknown') for r in recommendations]
        savings = [r.get('estimated_savings', 0) for r in recommendations]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=savings,
            y=names,
            orientation='h',
            marker=dict(
                color=savings,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Savings ($)")
            )
        ))
        
        fig.update_layout(
            title="Top Savings Opportunities",
            xaxis_title="Potential Monthly Savings ($)",
            yaxis_title="Recommendation",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="savings-chart")
    
    def _generate_executive_insights(self, metrics: ReportMetrics) -> List[Dict[str, str]]:
        """Generate executive insights."""
        insights = []
        
        # Spend insights
        if metrics.projected_spend > metrics.total_spend * 1.1:
            insights.append({
                'type': 'warning',
                'title': 'Projected Overspend',
                'message': f'Current spending trajectory will exceed budget by ${metrics.projected_spend - metrics.total_spend:,.0f}'
            })
        
        # Optimization insights
        if metrics.optimization_score < 70:
            insights.append({
                'type': 'opportunity',
                'title': 'Optimization Opportunity',
                'message': f'Current optimization score is {metrics.optimization_score:.1f}%. Implementing recommendations could save ${metrics.potential_savings:,.0f}/month'
            })
        
        # Waste insights
        if metrics.waste_percentage > 15:
            insights.append({
                'type': 'alert',
                'title': 'High Waste Detected',
                'message': f'{metrics.waste_percentage:.1f}% of spend is on unused or idle resources'
            })
        
        # Compliance insights
        if metrics.compliance_score < 80:
            insights.append({
                'type': 'compliance',
                'title': 'Compliance Issues',
                'message': f'Tag compliance is at {metrics.compliance_score:.1f}%. This affects cost allocation accuracy'
            })
        
        # Forecast insights
        if metrics.forecast_accuracy < 85:
            insights.append({
                'type': 'info',
                'title': 'Forecast Accuracy',
                'message': f'Cost forecast accuracy is {metrics.forecast_accuracy:.1f}%. Consider adjusting prediction models'
            })
        
        return insights
    
    def _get_report_period(self) -> Dict[str, str]:
        """Get report period information."""
        now = datetime.utcnow()
        month_start = now.replace(day=1)
        
        return {
            'start': month_start.strftime('%Y-%m-%d'),
            'end': now.strftime('%Y-%m-%d'),
            'month': now.strftime('%B %Y'),
            'days_elapsed': now.day,
            'days_remaining': 30 - now.day
        }
    
    def generate_cost_analysis_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed cost analysis report."""
        metrics = self._calculate_metrics(data)
        
        # Detailed cost breakdown
        cost_analysis = {
            'service_costs': self._analyze_service_costs(data),
            'account_costs': self._analyze_account_costs(data),
            'tag_costs': self._analyze_tag_costs(data),
            'region_costs': self._analyze_region_costs(data),
            'cost_anomalies': self._detect_cost_anomalies(data),
            'cost_drivers': self._identify_cost_drivers(data)
        }
        
        return {
            'metrics': metrics,
            'analysis': cost_analysis,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _analyze_service_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze costs by service."""
        costs = data.get('costs', {})
        service_data = {}
        
        for item in costs.get('items', []):
            service = item.get('service', 'Unknown')
            if service not in service_data:
                service_data[service] = {
                    'total_cost': 0,
                    'resource_count': 0,
                    'daily_average': 0,
                    'trend': [],
                    'top_resources': []
                }
            
            service_data[service]['total_cost'] += item.get('cost', 0)
            service_data[service]['resource_count'] += 1
        
        # Calculate averages and trends
        for service in service_data:
            data_point = service_data[service]
            data_point['daily_average'] = data_point['total_cost'] / 30
            data_point['cost_per_resource'] = (
                data_point['total_cost'] / data_point['resource_count']
                if data_point['resource_count'] > 0 else 0
            )
        
        return service_data
    
    def _analyze_account_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze costs by account."""
        costs = data.get('costs', {})
        account_data = {}
        
        for item in costs.get('items', []):
            account = item.get('account_id', 'Unknown')
            if account not in account_data:
                account_data[account] = {
                    'total_cost': 0,
                    'services': {},
                    'trend': []
                }
            
            account_data[account]['total_cost'] += item.get('cost', 0)
            
            service = item.get('service', 'Unknown')
            if service not in account_data[account]['services']:
                account_data[account]['services'][service] = 0
            account_data[account]['services'][service] += item.get('cost', 0)
        
        return account_data
    
    def _analyze_tag_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze costs by tags."""
        costs = data.get('costs', {})
        tag_data = {}
        
        for item in costs.get('items', []):
            tags = item.get('tags', {})
            
            for tag_key, tag_value in tags.items():
                if tag_key not in tag_data:
                    tag_data[tag_key] = {}
                
                if tag_value not in tag_data[tag_key]:
                    tag_data[tag_key][tag_value] = {
                        'total_cost': 0,
                        'resource_count': 0
                    }
                
                tag_data[tag_key][tag_value]['total_cost'] += item.get('cost', 0)
                tag_data[tag_key][tag_value]['resource_count'] += 1
        
        return tag_data
    
    def _analyze_region_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze costs by region."""
        costs = data.get('costs', {})
        region_data = {}
        
        for item in costs.get('items', []):
            region = item.get('region', 'Unknown')
            if region not in region_data:
                region_data[region] = {
                    'total_cost': 0,
                    'services': {},
                    'data_transfer': 0
                }
            
            region_data[region]['total_cost'] += item.get('cost', 0)
            
            service = item.get('service', 'Unknown')
            if service not in region_data[region]['services']:
                region_data[region]['services'][service] = 0
            region_data[region]['services'][service] += item.get('cost', 0)
            
            # Track data transfer costs
            if 'transfer' in item.get('usage_type', '').lower():
                region_data[region]['data_transfer'] += item.get('cost', 0)
        
        return region_data
    
    def _detect_cost_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect cost anomalies."""
        anomalies = []
        costs = data.get('costs', {})
        
        # Get daily costs
        daily_costs = []
        for item in costs.get('daily_costs', []):
            daily_costs.append(item.get('cost', 0))
        
        if len(daily_costs) > 7:
            # Calculate statistics
            mean = np.mean(daily_costs)
            std = np.std(daily_costs)
            
            # Detect anomalies (costs > 2 standard deviations from mean)
            for i, cost in enumerate(daily_costs):
                if abs(cost - mean) > 2 * std:
                    anomalies.append({
                        'date': (datetime.utcnow() - timedelta(days=len(daily_costs)-i-1)).strftime('%Y-%m-%d'),
                        'cost': cost,
                        'expected': mean,
                        'deviation': abs(cost - mean) / std,
                        'type': 'spike' if cost > mean else 'drop'
                    })
        
        return anomalies
    
    def _identify_cost_drivers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify main cost drivers."""
        drivers = []
        costs = data.get('costs', {})
        
        # Analyze by multiple dimensions
        dimensions = ['service', 'resource_type', 'region', 'account']
        
        for dimension in dimensions:
            dimension_costs = {}
            
            for item in costs.get('items', []):
                key = item.get(dimension, 'Unknown')
                dimension_costs[key] = dimension_costs.get(key, 0) + item.get('cost', 0)
            
            # Get top drivers
            sorted_costs = sorted(dimension_costs.items(), key=lambda x: x[1], reverse=True)
            total = sum(c for _, c in sorted_costs)
            
            for name, cost in sorted_costs[:3]:
                drivers.append({
                    'dimension': dimension,
                    'name': name,
                    'cost': cost,
                    'percentage': (cost / total * 100) if total > 0 else 0
                })
        
        return sorted(drivers, key=lambda x: x['cost'], reverse=True)
    
    def generate_html_report(self, dashboard_data: Dict[str, Any], 
                            template_name: str = 'executive_summary') -> str:
        """Generate HTML report from dashboard data."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Prepare template context
        context = {
            'metrics': dashboard_data['metrics'],
            'report_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
            'cost_breakdown_chart': dashboard_data['charts']['cost_breakdown_chart'],
            'trend_chart': dashboard_data['charts']['trend_chart']
        }
        
        return template.render(**context)
    
    def export_to_pdf(self, html_content: str, filename: str) -> str:
        """Export HTML report to PDF."""
        # In production, use a library like weasyprint or pdfkit
        # For now, just save as HTML
        output_path = f"/tmp/{filename}.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def schedule_reports(self, recipients: List[str], frequency: str = 'weekly'):
        """Schedule automatic report generation and distribution."""
        # In production, integrate with scheduling service
        logger.info(f"Scheduled {frequency} reports for {recipients}")
        
        schedule_config = {
            'frequency': frequency,
            'recipients': recipients,
            'reports': ['executive_summary', 'cost_analysis'],
            'delivery_method': 'email',
            'next_run': self._calculate_next_run(frequency)
        }
        
        return schedule_config
    
    def _calculate_next_run(self, frequency: str) -> str:
        """Calculate next report run time."""
        now = datetime.utcnow()
        
        if frequency == 'daily':
            next_run = now + timedelta(days=1)
        elif frequency == 'weekly':
            next_run = now + timedelta(weeks=1)
        elif frequency == 'monthly':
            # First day of next month
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1)
            else:
                next_run = now.replace(month=now.month + 1, day=1)
        else:
            next_run = now + timedelta(days=1)
        
        return next_run.strftime('%Y-%m-%d %H:%M:%S')