# Clouptimizer 🚀

**Multi-cloud cost optimization tool with intelligent rule engine**

Clouptimizer is a comprehensive cloud cost optimization tool that analyzes resources across AWS, Azure, and GCP to identify savings opportunities and provide actionable recommendations.

## ✨ Features

- **Multi-Cloud Support**: Analyze AWS, Azure, and GCP resources in a unified interface
- **Intelligent Analysis**: Multiple optimization engines for comprehensive cost analysis
- **Rule Engine**: Extensible rule-based optimization with custom rule support
- **Beautiful Reports**: Generate HTML, PDF, and Excel reports with executive summaries
- **CLI & API**: Command-line interface and REST API for automation
- **Real-time Monitoring**: Track optimization progress and savings

## 🎯 Key Capabilities

### Resource Collection
- Parallel scanning across multiple regions
- Support for 15+ cloud services per provider
- Intelligent caching for improved performance
- Tag-based filtering and resource grouping

### Optimization Analysis
- **Rightsizing**: Identify over-provisioned resources
- **Idle Resources**: Detect and eliminate waste
- **Reserved Instances**: Maximize commitment discounts
- **Storage Optimization**: Optimize storage classes and lifecycle
- **Network Optimization**: Reduce data transfer costs
- **Scheduling**: Implement start/stop schedules

### Reporting & Visualization
- Executive dashboards with KPIs
- Detailed technical reports
- Cost trending and forecasting
- Implementation roadmaps
- ROI calculations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Tsahi-Elkayam/clouptimizer.git
cd clouptimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

# Install provider-specific dependencies
pip install -r requirements/aws.txt    # For AWS
pip install -r requirements/azure.txt  # For Azure
pip install -r requirements/gcp.txt    # For GCP
```

### Configuration

```bash
# Configure cloud providers
clouptimizer configure --provider aws

# Or configure all providers interactively
clouptimizer configure
```

### Basic Usage

```bash
# Quick optimization scan
clouptimizer quick --provider aws

# Step-by-step workflow
clouptimizer scan --provider aws --regions us-east-1,us-west-2
clouptimizer analyze --scan-file scan_results.json
clouptimizer report --analysis-file analysis.json --format html

# Execute optimizations (dry-run by default)
clouptimizer optimize --analysis-file analysis.json --phase 1
```

## 📖 Documentation

### CLI Commands

#### `scan`
Collect resources from cloud providers
```bash
clouptimizer scan [OPTIONS]

Options:
  -p, --provider [aws|azure|gcp|all]  Cloud provider to scan
  -r, --regions TEXT                   Regions to scan
  -s, --services TEXT                  Services to scan
  -o, --output PATH                    Output file
  -f, --format [json|yaml|table]       Output format
```

#### `analyze`
Analyze resources for optimization opportunities
```bash
clouptimizer analyze [OPTIONS]

Options:
  -s, --scan-file PATH                 Scan results file
  -t, --analysis-types TEXT            Types of analysis
  --threshold INTEGER                   Minimum savings threshold
  -o, --output PATH                    Output file
```

#### `optimize`
Execute optimization recommendations
```bash
clouptimizer optimize [OPTIONS]

Options:
  -a, --analysis-file PATH             Analysis results file
  --dry-run / --no-dry-run            Perform dry run
  --auto-approve                       Auto-approve changes
  --phase [1|2|3|all]                 Optimization phase
```

#### `report`
Generate optimization reports
```bash
clouptimizer report [OPTIONS]

Options:
  --scan-file PATH                     Scan results file
  --analysis-file PATH                 Analysis results file
  -f, --format [html|pdf|json|excel]  Report format
  -o, --output PATH                    Output file
  --template [executive|detailed]      Report template
```

## 🏗️ Architecture

```
clouptimizer/
├── src/
│   ├── core/           # Core business logic
│   ├── providers/      # Cloud provider implementations
│   ├── rules/          # Rule engine
│   ├── analysis/       # Analysis engines
│   ├── reporting/      # Report generation
│   ├── cli/            # CLI interface
│   └── api/            # REST API (optional)
├── config/             # Configuration files
├── tests/              # Test suite
└── docs/               # Documentation
```

## 🔧 Advanced Configuration

### Custom Rules

Create custom optimization rules in YAML:

```yaml
# config/rules/custom/my_rule.yaml
name: large_instance_low_cpu
description: Identify large instances with low CPU usage
conditions:
  - field: instance_type
    operator: in
    value: [m5.xlarge, m5.2xlarge, m5.4xlarge]
  - field: cpu_utilization
    operator: less_than
    value: 20
actions:
  - type: recommendation
    message: Consider downsizing to smaller instance type
  - type: savings_calculation
    formula: current_cost * 0.5
```

### Provider Configuration

```yaml
# ~/.clouptimizer/config.yaml
general:
  default_provider: aws
  parallel_workers: 10
  cache_ttl_hours: 4

providers:
  aws:
    enabled: true
    profile: production
    regions:
      - us-east-1
      - us-west-2
      - eu-west-1
    services:
      - ec2
      - rds
      - s3

optimization:
  min_savings_threshold: 50
  confidence_threshold: 70
  exclude_tags:
    - Environment: Development
```

## 📊 Sample Output

```
Cloud Resource Scanner
Provider: aws
Regions: us-east-1, us-west-2
Services: All available

✓ AWS: 247 resources

Optimization Opportunities
┌─────────┬─────────────────────────────┬────────────────┬──────────┬────────────┐
│ Type    │ Description                 │ Monthly Savings│ Priority │ Confidence │
├─────────┼─────────────────────────────┼────────────────┼──────────┼────────────┤
│ Rightsizing │ Downsize 15 EC2 instances│ $1,200.00     │ HIGH     │ 85%        │
│ Unused  │ Delete 23 EBS volumes       │ $450.00        │ HIGH     │ 100%       │
│ Reserved│ Purchase RIs for RDS        │ $2,000.00      │ CRITICAL │ 75%        │
└─────────┴─────────────────────────────┴────────────────┴──────────┴────────────┘

Total Monthly Savings: $3,650.00
Annual Savings: $43,800.00
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AWS, Azure, and GCP for their comprehensive APIs
- The open-source community for inspiration and tools
- Contributors and users for their valuable feedback

## 📧 Contact

- GitHub: [@Tsahi-Elkayam](https://github.com/Tsahi-Elkayam)
- Email: your.email@example.com

## 🚧 Roadmap

- [ ] Kubernetes cost optimization
- [ ] Multi-cloud cost comparison
- [ ] ML-based prediction models
- [ ] Slack/Teams integration
- [ ] Terraform cost estimation
- [ ] Real-time cost anomaly detection
- [ ] Mobile app for monitoring

---

**Start optimizing your cloud costs today with Clouptimizer!** 🎯