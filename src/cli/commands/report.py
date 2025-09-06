import click
from rich.console import Console
from pathlib import Path
import json
from datetime import datetime


@click.command()
@click.option('--scan-file', type=click.Path(exists=True), help='Scan results file')
@click.option('--analysis-file', type=click.Path(exists=True), help='Analysis results file')
@click.option('--format', '-f', type=click.Choice(['html', 'pdf', 'json', 'excel']), 
              default='html', help='Report format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--template', type=click.Choice(['executive', 'detailed', 'technical']), 
              default='executive', help='Report template')
@click.option('--title', help='Custom report title')
@click.pass_context
def report(ctx, scan_file, analysis_file, format, output, template, title):
    """
    Generate optimization reports
    
    Examples:
        clouptimizer report --analysis-file analysis.json --format html
        clouptimizer report -f pdf -o report.pdf --template executive
    """
    console = ctx.obj['console']
    
    console.print(f"\n[bold]Report Generator[/bold]")
    console.print(f"Format: [cyan]{format.upper()}[/cyan]")
    console.print(f"Template: [cyan]{template.capitalize()}[/cyan]")
    
    # Load data
    scan_data = {}
    analysis_data = {}
    
    if scan_file:
        with open(scan_file, 'r') as f:
            scan_data = json.load(f)
        console.print(f"✓ Loaded scan data from [green]{scan_file}[/green]")
    
    if analysis_file:
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        console.print(f"✓ Loaded analysis data from [green]{analysis_file}[/green]")
    
    if not scan_data and not analysis_data:
        console.print("[yellow]No data files provided, using sample data[/yellow]")
        scan_data, analysis_data = _get_sample_report_data()
    
    # Generate report
    with console.status(f"[bold green]Generating {format.upper()} report..."):
        if format == 'html':
            report_content = _generate_html_report(scan_data, analysis_data, template, title)
            file_ext = '.html'
        elif format == 'json':
            report_content = json.dumps({
                'scan': scan_data,
                'analysis': analysis_data,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'template': template
                }
            }, indent=2)
            file_ext = '.json'
        else:
            # Placeholder for other formats
            report_content = f"# {title or 'Cloud Optimization Report'}\n\nFormat not yet implemented"
            file_ext = f'.{format}'
    
    # Save report
    if not output:
        output = f"clouptimizer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
    
    with open(output, 'w') as f:
        f.write(report_content)
    
    console.print(f"\n✓ Report generated: [green]{output}[/green]")
    
    # Display summary
    if analysis_data:
        console.print(f"\n[bold]Report Summary:[/bold]")
        console.print(f"  Total optimizations: [cyan]{analysis_data.get('summary', {}).get('total_optimizations', 0)}[/cyan]")
        console.print(f"  Monthly savings: [green]${analysis_data.get('summary', {}).get('total_monthly_savings', 0):,.2f}[/green]")
        console.print(f"  Annual savings: [bold green]${analysis_data.get('summary', {}).get('total_annual_savings', 0):,.2f}[/bold green]")


def _generate_html_report(scan_data, analysis_data, template, custom_title):
    """Generate HTML report"""
    
    title = custom_title or "Cloud Cost Optimization Report"
    
    # Calculate summary metrics
    total_resources = scan_data.get('summary', {}).get('total_resources', 0)
    total_cost = scan_data.get('summary', {}).get('total_monthly_cost', 0)
    total_optimizations = analysis_data.get('summary', {}).get('total_optimizations', 0)
    monthly_savings = analysis_data.get('summary', {}).get('total_monthly_savings', 0)
    annual_savings = analysis_data.get('summary', {}).get('total_annual_savings', 0)
    
    # Generate optimization rows
    optimization_rows = ""
    for opt in analysis_data.get('optimizations', [])[:10]:  # Top 10
        optimization_rows += f"""
        <tr>
            <td>{opt.get('type', 'N/A').capitalize()}</td>
            <td>{opt.get('title', 'N/A')}</td>
            <td>${opt.get('monthly_savings', 0):,.2f}</td>
            <td><span class="priority-{opt.get('priority', 'low')}">{opt.get('priority', 'N/A').upper()}</span></td>
            <td>{opt.get('confidence', 0)}%</td>
        </tr>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
            padding: 2rem;
            background: #f8f9fa;
        }}
        
        .metric {{
            text-align: center;
            padding: 1.5rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            color: #666;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }}
        
        .content {{
            padding: 3rem;
        }}
        
        .section {{
            margin-bottom: 3rem;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th {{
            background: #f8f9fa;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 1px;
        }}
        
        td {{
            padding: 1rem;
            border-bottom: 1px solid #e9ecef;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .priority-critical {{
            background: #dc3545;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        
        .priority-high {{
            background: #fd7e14;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        
        .priority-medium {{
            background: #ffc107;
            color: #333;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        
        .priority-low {{
            background: #28a745;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 2rem;
            text-align: center;
            color: #666;
            font-size: 0.9rem;
        }}
        
        .chart-container {{
            margin: 2rem 0;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 1rem 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{total_resources}</div>
                <div class="metric-label">Total Resources</div>
            </div>
            <div class="metric">
                <div class="metric-value">${total_cost:,.0f}</div>
                <div class="metric-label">Monthly Cost</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_optimizations}</div>
                <div class="metric-label">Optimizations</div>
            </div>
            <div class="metric">
                <div class="metric-value">${monthly_savings:,.0f}</div>
                <div class="metric-label">Monthly Savings</div>
            </div>
            <div class="metric">
                <div class="metric-value">${annual_savings:,.0f}</div>
                <div class="metric-label">Annual Savings</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report provides a comprehensive analysis of your cloud infrastructure costs across all providers. 
                Our analysis has identified <strong>{total_optimizations} optimization opportunities</strong> that could save your organization 
                <strong>${monthly_savings:,.2f} per month</strong> ({(monthly_savings/total_cost*100):.1f}% reduction).</p>
                
                <div class="chart-container">
                    <h3>Potential Savings Impact</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {min((monthly_savings/total_cost*100), 100):.1f}%">
                            {(monthly_savings/total_cost*100):.1f}% Cost Reduction
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Top Optimization Opportunities</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Description</th>
                            <th>Monthly Savings</th>
                            <th>Priority</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {optimization_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Implementation Roadmap</h2>
                <ol>
                    <li><strong>Phase 1 - Quick Wins (Week 1-2):</strong> Implement high-confidence, low-risk optimizations such as deleting unused resources and stopping idle instances.</li>
                    <li><strong>Phase 2 - Rightsizing (Week 3-4):</strong> Resize over-provisioned resources based on actual utilization metrics.</li>
                    <li><strong>Phase 3 - Reserved Capacity (Month 2):</strong> Purchase reserved instances and savings plans for predictable workloads.</li>
                </ol>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Clouptimizer - Multi-cloud Cost Optimization Tool</p>
            <p>© 2024 Your Organization. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content


def _get_sample_report_data():
    """Get sample data for report generation"""
    scan_data = {
        'summary': {
            'total_resources': 250,
            'total_monthly_cost': 15000,
            'providers': {
                'aws': {'resources': 150, 'monthly_cost': 10000},
                'azure': {'resources': 75, 'monthly_cost': 4000},
                'gcp': {'resources': 25, 'monthly_cost': 1000}
            }
        }
    }
    
    analysis_data = {
        'summary': {
            'total_optimizations': 45,
            'total_monthly_savings': 4500,
            'total_annual_savings': 54000
        },
        'optimizations': [
            {
                'type': 'rightsizing',
                'title': 'Rightsize 15 EC2 instances',
                'monthly_savings': 1200,
                'priority': 'critical',
                'confidence': 85
            },
            {
                'type': 'unused',
                'title': 'Delete 23 unattached EBS volumes',
                'monthly_savings': 450,
                'priority': 'high',
                'confidence': 100
            },
            {
                'type': 'idle',
                'title': 'Stop 8 idle RDS instances',
                'monthly_savings': 800,
                'priority': 'high',
                'confidence': 95
            },
            {
                'type': 'reserved',
                'title': 'Purchase Reserved Instances',
                'monthly_savings': 2050,
                'priority': 'critical',
                'confidence': 75
            }
        ]
    }
    
    return scan_data, analysis_data