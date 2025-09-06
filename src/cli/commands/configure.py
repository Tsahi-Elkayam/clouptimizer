import click
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
import yaml
import json
from pathlib import Path


@click.command()
@click.option('--provider', type=click.Choice(['aws', 'azure', 'gcp', 'all']),
              help='Configure specific provider')
@click.option('--profile', help='Profile name to configure')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--reset', is_flag=True, help='Reset configuration to defaults')
@click.pass_context
def configure(ctx, provider, profile, show, reset):
    """
    Configure cloud provider credentials and settings
    
    Examples:
        clouptimizer configure --provider aws
        clouptimizer configure --show
        clouptimizer configure --profile production
    """
    console = ctx.obj['console']
    config_file = ctx.obj['config_file']
    
    console.print(f"\n[bold]Configuration Manager[/bold]")
    
    # Ensure config directory exists
    config_dir = Path(config_file).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing configuration
    config = _load_config(config_file)
    
    if show:
        _show_configuration(console, config)
        return
    
    if reset:
        if Confirm.ask("[yellow]Are you sure you want to reset all configurations?[/yellow]"):
            config = _get_default_config()
            _save_config(config_file, config)
            console.print("[green]✓ Configuration reset to defaults[/green]")
        return
    
    # Configure specific provider or all
    if provider == 'all':
        providers_to_configure = ['aws', 'azure', 'gcp']
    elif provider:
        providers_to_configure = [provider]
    else:
        # Interactive mode - ask which providers to configure
        console.print("\nWhich providers would you like to configure?")
        providers_to_configure = []
        
        if Confirm.ask("  Configure AWS?"):
            providers_to_configure.append('aws')
        if Confirm.ask("  Configure Azure?"):
            providers_to_configure.append('azure')
        if Confirm.ask("  Configure GCP?"):
            providers_to_configure.append('gcp')
    
    # Configure selected providers
    for prov in providers_to_configure:
        console.print(f"\n[cyan]Configuring {prov.upper()}...[/cyan]")
        config['providers'][prov] = _configure_provider(console, prov, config.get('providers', {}).get(prov, {}))
    
    # Configure general settings
    if Confirm.ask("\n[yellow]Configure general settings?[/yellow]"):
        config['general'] = _configure_general_settings(console, config.get('general', {}))
    
    # Save configuration
    _save_config(config_file, config)
    console.print(f"\n[green]✓ Configuration saved to {config_file}[/green]")
    
    # Display summary
    _show_configuration_summary(console, config)


def _load_config(config_file):
    """Load configuration from file"""
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f) or {}
    return _get_default_config()


def _save_config(config_file, config):
    """Save configuration to file"""
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def _get_default_config():
    """Get default configuration"""
    return {
        'general': {
            'default_provider': 'aws',
            'parallel_workers': 10,
            'cache_enabled': True,
            'cache_ttl_hours': 4,
            'output_format': 'table',
            'auto_confirm': False
        },
        'providers': {
            'aws': {
                'enabled': False,
                'profile': 'default',
                'regions': ['us-east-1', 'us-west-2'],
                'services': []  # Empty means all services
            },
            'azure': {
                'enabled': False,
                'subscription_id': '',
                'tenant_id': '',
                'regions': [],
                'services': []
            },
            'gcp': {
                'enabled': False,
                'project_id': '',
                'credentials_file': '',
                'regions': [],
                'services': []
            }
        },
        'optimization': {
            'min_savings_threshold': 10,
            'confidence_threshold': 50,
            'exclude_tags': [],
            'include_tags': []
        },
        'reporting': {
            'default_template': 'executive',
            'include_charts': True,
            'email_enabled': False,
            'email_recipients': []
        }
    }


def _configure_provider(console, provider, existing_config):
    """Configure a specific provider"""
    config = existing_config.copy()
    
    if provider == 'aws':
        config['enabled'] = Confirm.ask(f"  Enable AWS?", default=existing_config.get('enabled', False))
        
        if config['enabled']:
            config['profile'] = Prompt.ask("  AWS Profile", default=existing_config.get('profile', 'default'))
            
            regions_input = Prompt.ask("  Regions (comma-separated)", 
                                      default=','.join(existing_config.get('regions', ['us-east-1'])))
            config['regions'] = [r.strip() for r in regions_input.split(',')]
            
            if Confirm.ask("  Configure specific services? (default: all)"):
                services_input = Prompt.ask("    Services (comma-separated, e.g., ec2,s3,rds)")
                config['services'] = [s.strip() for s in services_input.split(',')]
            else:
                config['services'] = []
    
    elif provider == 'azure':
        config['enabled'] = Confirm.ask(f"  Enable Azure?", default=existing_config.get('enabled', False))
        
        if config['enabled']:
            config['subscription_id'] = Prompt.ask("  Subscription ID", 
                                                  default=existing_config.get('subscription_id', ''))
            config['tenant_id'] = Prompt.ask("  Tenant ID", 
                                            default=existing_config.get('tenant_id', ''))
            
            regions_input = Prompt.ask("  Regions (comma-separated)", 
                                      default=','.join(existing_config.get('regions', ['eastus'])))
            config['regions'] = [r.strip() for r in regions_input.split(',')]
    
    elif provider == 'gcp':
        config['enabled'] = Confirm.ask(f"  Enable GCP?", default=existing_config.get('enabled', False))
        
        if config['enabled']:
            config['project_id'] = Prompt.ask("  Project ID", 
                                             default=existing_config.get('project_id', ''))
            config['credentials_file'] = Prompt.ask("  Credentials file path", 
                                                   default=existing_config.get('credentials_file', ''))
            
            regions_input = Prompt.ask("  Regions (comma-separated)", 
                                      default=','.join(existing_config.get('regions', ['us-central1'])))
            config['regions'] = [r.strip() for r in regions_input.split(',')]
    
    return config


def _configure_general_settings(console, existing_config):
    """Configure general settings"""
    config = existing_config.copy()
    
    config['default_provider'] = Prompt.ask("Default provider", 
                                           choices=['aws', 'azure', 'gcp'],
                                           default=existing_config.get('default_provider', 'aws'))
    
    config['parallel_workers'] = int(Prompt.ask("Parallel workers", 
                                               default=str(existing_config.get('parallel_workers', 10))))
    
    config['cache_enabled'] = Confirm.ask("Enable caching?", 
                                         default=existing_config.get('cache_enabled', True))
    
    if config['cache_enabled']:
        config['cache_ttl_hours'] = int(Prompt.ask("Cache TTL (hours)", 
                                                  default=str(existing_config.get('cache_ttl_hours', 4))))
    
    config['output_format'] = Prompt.ask("Default output format", 
                                        choices=['table', 'json', 'yaml'],
                                        default=existing_config.get('output_format', 'table'))
    
    config['auto_confirm'] = Confirm.ask("Auto-confirm actions?", 
                                        default=existing_config.get('auto_confirm', False))
    
    return config


def _show_configuration(console, config):
    """Display current configuration"""
    console.print("\n[bold]Current Configuration:[/bold]\n")
    
    # Format configuration as YAML for display
    config_yaml = yaml.dump(config, default_flow_style=False)
    
    console.print(Panel(config_yaml, title="Configuration", border_style="cyan"))


def _show_configuration_summary(console, config):
    """Show configuration summary"""
    enabled_providers = [
        prov for prov, cfg in config.get('providers', {}).items()
        if cfg.get('enabled', False)
    ]
    
    summary = f"""[bold]Configuration Summary:[/bold]
    
Enabled Providers: {', '.join(enabled_providers) if enabled_providers else 'None'}
Default Provider: {config.get('general', {}).get('default_provider', 'Not set')}
Parallel Workers: {config.get('general', {}).get('parallel_workers', 10)}
Cache Enabled: {config.get('general', {}).get('cache_enabled', False)}
Output Format: {config.get('general', {}).get('output_format', 'table')}
    """
    
    console.print("\n")
    console.print(Panel(summary, border_style="green"))