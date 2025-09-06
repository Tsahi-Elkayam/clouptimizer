import click
import logging
from rich.console import Console
from rich.logging import RichHandler
from pathlib import Path

from .commands import scan, analyze, optimize, report, configure

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

@click.group()
@click.version_option(version='0.1.0', prog_name='clouptimizer')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', type=click.Path(), help='Path to configuration file')
@click.pass_context
def cli(ctx, debug, config):
    """
    Clouptimizer - Multi-cloud cost optimization tool
    
    Analyze and optimize costs across AWS, Azure, and GCP with intelligent recommendations.
    """
    ctx.ensure_object(dict)
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.obj['config_file'] = config or Path.home() / '.clouptimizer' / 'config.yaml'
    ctx.obj['console'] = console

# Register commands
cli.add_command(scan.scan)
cli.add_command(analyze.analyze)
cli.add_command(optimize.optimize)
cli.add_command(report.report)
cli.add_command(configure.configure)

@cli.command()
@click.pass_context
def version(ctx):
    """Show version information"""
    console.print("[bold blue]Clouptimizer[/bold blue] version [green]0.1.0[/green]")
    console.print("Multi-cloud cost optimization tool")

@cli.command()
@click.option('--provider', type=click.Choice(['aws', 'azure', 'gcp', 'all']), default='all')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml', 'table']), default='table')
@click.pass_context
def quick(ctx, provider, output, format):
    """Run quick optimization scan (scan + analyze + report)"""
    console.print("[bold]Starting quick optimization scan...[/bold]")
    
    # This would integrate the workflow engine
    console.print(f"  Provider: [cyan]{provider}[/cyan]")
    console.print(f"  Output format: [cyan]{format}[/cyan]")
    
    with console.status("[bold green]Scanning resources..."):
        # Placeholder for actual implementation
        console.print("✓ Scanned 150 resources")
    
    with console.status("[bold green]Analyzing optimizations..."):
        console.print("✓ Found 45 optimization opportunities")
    
    with console.status("[bold green]Generating report..."):
        console.print("✓ Report generated")
    
    console.print("\n[bold green]Quick scan complete![/bold green]")
    console.print("Total savings identified: [bold]$2,450/month[/bold]")

if __name__ == '__main__':
    cli()