import click
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
import json


@click.command()
@click.option('--analysis-file', '-a', type=click.Path(exists=True),
              help='Path to analysis results file')
@click.option('--dry-run', is_flag=True, default=True, help='Perform dry run (no actual changes)')
@click.option('--auto-approve', is_flag=True, help='Auto-approve all optimizations')
@click.option('--phase', type=click.Choice(['1', '2', '3', 'all']), default='1',
              help='Optimization phase to execute')
@click.option('--output', '-o', type=click.Path(), help='Output file for execution plan')
@click.pass_context
def optimize(ctx, analysis_file, dry_run, auto_approve, phase, output):
    """
    Execute optimization recommendations
    
    Examples:
        clouptimizer optimize --analysis-file analysis.json --dry-run
        clouptimizer optimize -a analysis.json --phase 1 --auto-approve
    """
    console = ctx.obj['console']
    
    console.print(f"\n[bold]Optimization Executor[/bold]")
    console.print(f"Mode: [{'yellow' if dry_run else 'red'}]{'DRY RUN' if dry_run else 'LIVE'}[/{'yellow' if dry_run else 'red'}]")
    
    # Load analysis results
    if analysis_file:
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        console.print(f"✓ Loaded {analysis_data['summary']['total_optimizations']} optimizations")
    else:
        console.print("[yellow]No analysis file provided, using sample data[/yellow]")
        analysis_data = _get_sample_analysis_data()
    
    # Create execution plan
    execution_plan = _create_execution_plan(analysis_data, phase)
    
    # Display plan
    _display_execution_plan(console, execution_plan)
    
    # Confirm execution
    if not auto_approve and not dry_run:
        if not Confirm.ask("\n[yellow]Do you want to proceed with these optimizations?[/yellow]"):
            console.print("[red]Optimization cancelled[/red]")
            return
    
    # Execute optimizations
    if dry_run:
        console.print("\n[yellow]DRY RUN - No actual changes will be made[/yellow]")
    
    results = _execute_optimizations(console, execution_plan, dry_run)
    
    # Display results
    _display_execution_results(console, results)
    
    # Save execution plan if requested
    if output:
        with open(output, 'w') as f:
            json.dump({
                'plan': execution_plan,
                'results': results
            }, f, indent=2)
        console.print(f"\n✓ Execution plan saved to [green]{output}[/green]")


def _create_execution_plan(analysis_data, phase):
    """Create execution plan from analysis data"""
    
    plan = {
        'phase': phase,
        'optimizations': [],
        'total_savings': 0,
        'total_actions': 0
    }
    
    # Group optimizations by complexity/phase
    for opt in analysis_data.get('optimizations', []):
        opt_phase = _determine_phase(opt)
        
        if phase == 'all' or phase == str(opt_phase):
            plan['optimizations'].append({
                'id': opt['id'],
                'type': opt['type'],
                'title': opt['title'],
                'resource': opt.get('resource', 'N/A'),
                'savings': opt['monthly_savings'],
                'priority': opt['priority'],
                'actions': _generate_actions(opt)
            })
            plan['total_savings'] += opt['monthly_savings']
            plan['total_actions'] += len(_generate_actions(opt))
    
    return plan


def _determine_phase(optimization):
    """Determine which phase an optimization belongs to"""
    
    # Phase 1: Quick wins (unused, idle)
    if optimization['type'] in ['unused', 'idle']:
        return 1
    
    # Phase 2: Standard optimizations (rightsizing)
    elif optimization['type'] in ['rightsizing', 'scheduling']:
        return 2
    
    # Phase 3: Complex (reserved, migration)
    else:
        return 3


def _generate_actions(optimization):
    """Generate actions for an optimization"""
    
    actions = []
    
    if optimization['type'] == 'unused':
        actions.append({
            'type': 'delete',
            'description': f"Delete {optimization.get('resource', 'resource')}"
        })
    
    elif optimization['type'] == 'idle':
        actions.append({
            'type': 'stop',
            'description': f"Stop {optimization.get('resource', 'resource')}"
        })
    
    elif optimization['type'] == 'rightsizing':
        actions.append({
            'type': 'resize',
            'description': f"Resize {optimization.get('resource', 'resource')}"
        })
    
    elif optimization['type'] == 'reserved':
        actions.append({
            'type': 'purchase',
            'description': "Purchase reserved capacity"
        })
    
    return actions


def _execute_optimizations(console, plan, dry_run):
    """Execute the optimization plan"""
    
    results = {
        'executed': 0,
        'succeeded': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
    with console.status("[bold green]Executing optimizations...") as status:
        for opt in plan['optimizations']:
            status.update(f"Executing: {opt['title']}")
            
            # Simulate execution
            if dry_run:
                result = {
                    'id': opt['id'],
                    'status': 'dry_run',
                    'message': 'Would execute optimization'
                }
                results['skipped'] += 1
            else:
                # In real implementation, this would execute actual changes
                result = {
                    'id': opt['id'],
                    'status': 'success',
                    'message': 'Optimization executed successfully'
                }
                results['succeeded'] += 1
            
            results['executed'] += 1
            results['details'].append(result)
    
    return results


def _display_execution_plan(console, plan):
    """Display the execution plan"""
    
    table = Table(title=f"Execution Plan - Phase {plan['phase']}", 
                  show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Actions", justify="center", style="yellow")
    table.add_column("Savings", justify="right", style="green")
    
    for opt in plan['optimizations']:
        table.add_row(
            opt['id'],
            opt['type'].capitalize(),
            opt['title'],
            str(len(opt['actions'])),
            f"${opt['savings']:,.2f}/mo"
        )
    
    console.print("\n")
    console.print(table)
    
    console.print(f"\nTotal optimizations: [bold]{len(plan['optimizations'])}[/bold]")
    console.print(f"Total actions: [bold]{plan['total_actions']}[/bold]")
    console.print(f"Total monthly savings: [bold green]${plan['total_savings']:,.2f}[/bold green]")


def _display_execution_results(console, results):
    """Display execution results"""
    
    summary = f"""[bold]Execution Results[/bold]
    
Executed: [cyan]{results['executed']}[/cyan]
Succeeded: [green]{results['succeeded']}[/green]
Failed: [red]{results['failed']}[/red]
Skipped: [yellow]{results['skipped']}[/yellow]
    """
    
    console.print("\n")
    console.print(Panel(summary, border_style="green" if results['failed'] == 0 else "red"))


def _get_sample_analysis_data():
    """Get sample analysis data"""
    return {
        'summary': {
            'total_optimizations': 5,
            'total_monthly_savings': 1000
        },
        'optimizations': [
            {
                'id': 'opt_001',
                'type': 'unused',
                'resource': 'vol-123',
                'title': 'Delete unattached volume',
                'monthly_savings': 50,
                'priority': 'high'
            },
            {
                'id': 'opt_002',
                'type': 'idle',
                'resource': 'i-456',
                'title': 'Stop idle instance',
                'monthly_savings': 100,
                'priority': 'high'
            }
        ]
    }