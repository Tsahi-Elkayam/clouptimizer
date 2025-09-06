"""
Multi-Account Support with AWS Organizations
Manages multiple AWS accounts, cross-account access, and consolidated billing.
"""

import logging
import boto3
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import concurrent.futures
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AccountStatus(str, Enum):
    """AWS Account status"""
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    PENDING = "PENDING"
    INACTIVE = "INACTIVE"


class AccountType(str, Enum):
    """Account type in organization"""
    MANAGEMENT = "management"
    MEMBER = "member"
    DELEGATED_ADMIN = "delegated_admin"
    LINKED = "linked"


@dataclass
class AWSAccount:
    """Represents an AWS account in the organization"""
    account_id: str
    account_name: str
    email: str
    status: AccountStatus
    account_type: AccountType
    organizational_unit: str
    tags: Dict[str, str]
    joined_date: Optional[datetime] = None
    parent_account_id: Optional[str] = None
    assumed_role_arn: Optional[str] = None
    cross_account_role: Optional[str] = "OrganizationAccountAccessRole"
    regions: List[str] = field(default_factory=lambda: ['us-east-1'])
    services: List[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        return self.status == AccountStatus.ACTIVE
    
    @property
    def role_arn(self) -> str:
        """Get the role ARN for cross-account access"""
        if self.assumed_role_arn:
            return self.assumed_role_arn
        return f"arn:aws:iam::{self.account_id}:role/{self.cross_account_role}"


@dataclass
class OrganizationalUnit:
    """Represents an Organizational Unit"""
    ou_id: str
    name: str
    parent_id: Optional[str]
    accounts: List[str]
    child_ous: List[str]
    policies: List[str]
    tags: Dict[str, str]
    
    @property
    def account_count(self) -> int:
        return len(self.accounts)


@dataclass
class ConsolidatedBilling:
    """Consolidated billing information"""
    billing_account_id: str
    linked_accounts: List[str]
    total_cost: float
    by_account_costs: Dict[str, float]
    by_service_costs: Dict[str, float]
    billing_period: str
    currency: str = "USD"
    credits: float = 0
    taxes: float = 0
    
    @property
    def net_cost(self) -> float:
        return self.total_cost - self.credits + self.taxes


class MultiAccountManager:
    """
    Manages multi-account AWS environments using Organizations.
    Provides cross-account access and consolidated cost analysis.
    """
    
    def __init__(self, master_session=None):
        """
        Initialize multi-account manager.
        
        Args:
            master_session: Boto3 session for management account
        """
        self.master_session = master_session or boto3.Session()
        self.organizations_client = self.master_session.client('organizations')
        self.sts_client = self.master_session.client('sts')
        self.accounts: List[AWSAccount] = []
        self.organizational_units: List[OrganizationalUnit] = []
        self.account_sessions: Dict[str, boto3.Session] = {}
        self.billing_data: Optional[ConsolidatedBilling] = None
    
    def discover_accounts(self) -> List[AWSAccount]:
        """
        Discover all accounts in the AWS Organization.
        
        Returns:
            List of AWS accounts
        """
        self.accounts = []
        
        try:
            # Get organization information
            org_info = self.organizations_client.describe_organization()
            master_account_id = org_info['Organization']['MasterAccountId']
            
            # List all accounts
            paginator = self.organizations_client.get_paginator('list_accounts')
            
            for page in paginator.paginate():
                for account in page.get('Accounts', []):
                    # Get account details
                    account_obj = self._create_account_object(account, master_account_id)
                    
                    # Get account tags
                    try:
                        tags_response = self.organizations_client.list_tags_for_resource(
                            ResourceId=account['Id']
                        )
                        account_obj.tags = {
                            tag['Key']: tag['Value'] 
                            for tag in tags_response.get('Tags', [])
                        }
                    except:
                        pass
                    
                    self.accounts.append(account_obj)
            
            logger.info(f"Discovered {len(self.accounts)} AWS accounts")
            
            # Discover organizational units
            self._discover_organizational_units()
            
        except ClientError as e:
            logger.error(f"Error discovering accounts: {e}")
            
            # Fallback to single account mode
            current_account = self._get_current_account()
            if current_account:
                self.accounts = [current_account]
        
        return self.accounts
    
    def _create_account_object(self, account_data: Dict, master_id: str) -> AWSAccount:
        """Create AWSAccount object from API data"""
        account_type = AccountType.MANAGEMENT if account_data['Id'] == master_id else AccountType.MEMBER
        
        return AWSAccount(
            account_id=account_data['Id'],
            account_name=account_data.get('Name', 'Unknown'),
            email=account_data.get('Email', ''),
            status=AccountStatus(account_data.get('Status', 'ACTIVE')),
            account_type=account_type,
            organizational_unit=self._get_account_ou(account_data['Id']),
            tags={},
            joined_date=account_data.get('JoinedTimestamp'),
            parent_account_id=master_id if account_type == AccountType.MEMBER else None
        )
    
    def _get_current_account(self) -> Optional[AWSAccount]:
        """Get current account as fallback"""
        try:
            identity = self.sts_client.get_caller_identity()
            return AWSAccount(
                account_id=identity['Account'],
                account_name="Current Account",
                email="",
                status=AccountStatus.ACTIVE,
                account_type=AccountType.MANAGEMENT,
                organizational_unit="root",
                tags={}
            )
        except:
            return None
    
    def _discover_organizational_units(self):
        """Discover organizational units structure"""
        self.organizational_units = []
        
        try:
            # Get root OU
            roots = self.organizations_client.list_roots()
            
            for root in roots.get('Roots', []):
                self._traverse_ou(root['Id'], None)
                
        except ClientError as e:
            logger.error(f"Error discovering OUs: {e}")
    
    def _traverse_ou(self, ou_id: str, parent_id: Optional[str]):
        """Recursively traverse organizational units"""
        try:
            # Get OU details
            if ou_id.startswith('r-'):  # Root
                ou_name = "Root"
            else:
                ou_details = self.organizations_client.describe_organizational_unit(
                    OrganizationalUnitId=ou_id
                )
                ou_name = ou_details['OrganizationalUnit']['Name']
            
            # List accounts in OU
            accounts = []
            paginator = self.organizations_client.get_paginator('list_accounts_for_parent')
            for page in paginator.paginate(ParentId=ou_id):
                accounts.extend([acc['Id'] for acc in page.get('Accounts', [])])
            
            # List child OUs
            child_ous = []
            ou_paginator = self.organizations_client.get_paginator('list_organizational_units_for_parent')
            for page in ou_paginator.paginate(ParentId=ou_id):
                for child_ou in page.get('OrganizationalUnits', []):
                    child_ous.append(child_ou['Id'])
                    # Recursively traverse children
                    self._traverse_ou(child_ou['Id'], ou_id)
            
            # Get policies
            policies = []
            try:
                policy_response = self.organizations_client.list_policies_for_target(
                    TargetId=ou_id,
                    Filter='SERVICE_CONTROL_POLICY'
                )
                policies = [p['Id'] for p in policy_response.get('Policies', [])]
            except:
                pass
            
            # Create OU object
            ou = OrganizationalUnit(
                ou_id=ou_id,
                name=ou_name,
                parent_id=parent_id,
                accounts=accounts,
                child_ous=child_ous,
                policies=policies,
                tags={}
            )
            
            self.organizational_units.append(ou)
            
        except ClientError as e:
            logger.error(f"Error traversing OU {ou_id}: {e}")
    
    def _get_account_ou(self, account_id: str) -> str:
        """Get organizational unit for account"""
        try:
            parents = self.organizations_client.list_parents(ChildId=account_id)
            if parents.get('Parents'):
                return parents['Parents'][0]['Id']
        except:
            pass
        return "unknown"
    
    def assume_role(self, account: AWSAccount) -> Optional[boto3.Session]:
        """
        Assume role in member account for cross-account access.
        
        Args:
            account: AWS account to access
            
        Returns:
            Boto3 session for the account or None if failed
        """
        if account.account_id in self.account_sessions:
            return self.account_sessions[account.account_id]
        
        try:
            # Assume role
            response = self.sts_client.assume_role(
                RoleArn=account.role_arn,
                RoleSessionName=f"Clouptimizer-{account.account_id}"
            )
            
            # Create session with temporary credentials
            credentials = response['Credentials']
            session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            
            self.account_sessions[account.account_id] = session
            logger.info(f"Successfully assumed role in account {account.account_id}")
            
            return session
            
        except ClientError as e:
            logger.error(f"Failed to assume role in account {account.account_id}: {e}")
            return None
    
    def collect_multi_account_resources(self, collector_func, 
                                       accounts: Optional[List[AWSAccount]] = None,
                                       parallel: bool = True) -> Dict[str, List[Dict]]:
        """
        Collect resources from multiple accounts.
        
        Args:
            collector_func: Function to collect resources (takes session as argument)
            accounts: List of accounts to collect from (None = all active)
            parallel: Whether to collect in parallel
            
        Returns:
            Dictionary of resources by account ID
        """
        if not accounts:
            accounts = [acc for acc in self.accounts if acc.is_active]
        
        resources_by_account = {}
        
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                
                for account in accounts:
                    session = self.assume_role(account)
                    if session:
                        future = executor.submit(collector_func, session)
                        futures[future] = account.account_id
                
                for future in concurrent.futures.as_completed(futures):
                    account_id = futures[future]
                    try:
                        resources = future.result()
                        resources_by_account[account_id] = resources
                        logger.info(f"Collected {len(resources)} resources from account {account_id}")
                    except Exception as e:
                        logger.error(f"Error collecting from account {account_id}: {e}")
                        resources_by_account[account_id] = []
        else:
            for account in accounts:
                session = self.assume_role(account)
                if session:
                    try:
                        resources = collector_func(session)
                        resources_by_account[account.account_id] = resources
                        logger.info(f"Collected {len(resources)} resources from account {account.account_id}")
                    except Exception as e:
                        logger.error(f"Error collecting from account {account.account_id}: {e}")
                        resources_by_account[account.account_id] = []
        
        return resources_by_account
    
    def get_consolidated_billing(self, start_date: str, end_date: str) -> ConsolidatedBilling:
        """
        Get consolidated billing information across all accounts.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Consolidated billing information
        """
        try:
            ce_client = self.master_session.client('ce', region_name='us-east-1')
            
            # Get cost and usage
            response = ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'LINKED_ACCOUNT'},
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
            # Process billing data
            by_account = defaultdict(float)
            by_service = defaultdict(float)
            total_cost = 0
            
            for result in response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    account = group['Keys'][0]
                    service = group['Keys'][1]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    
                    by_account[account] += cost
                    by_service[service] += cost
                    total_cost += cost
            
            # Get master account ID
            master_account = next((acc.account_id for acc in self.accounts 
                                 if acc.account_type == AccountType.MANAGEMENT), 
                                'unknown')
            
            self.billing_data = ConsolidatedBilling(
                billing_account_id=master_account,
                linked_accounts=list(by_account.keys()),
                total_cost=total_cost,
                by_account_costs=dict(by_account),
                by_service_costs=dict(by_service),
                billing_period=f"{start_date} to {end_date}"
            )
            
            return self.billing_data
            
        except ClientError as e:
            logger.error(f"Error getting consolidated billing: {e}")
            return ConsolidatedBilling(
                billing_account_id="unknown",
                linked_accounts=[],
                total_cost=0,
                by_account_costs={},
                by_service_costs={},
                billing_period=f"{start_date} to {end_date}"
            )
    
    def apply_organization_policy(self, policy_name: str, 
                                 target_ou: Optional[str] = None,
                                 policy_type: str = "tag_policy") -> bool:
        """
        Apply organization-wide policy.
        
        Args:
            policy_name: Name of the policy
            target_ou: Target OU (None = root)
            policy_type: Type of policy (tag_policy, scp, etc.)
            
        Returns:
            Success status
        """
        try:
            # Create or update policy
            policy_content = self._generate_policy_content(policy_name, policy_type)
            
            # Check if policy exists
            policies = self.organizations_client.list_policies(Filter=policy_type.upper())
            existing_policy = next((p for p in policies['Policies'] 
                                  if p['Name'] == policy_name), None)
            
            if existing_policy:
                policy_id = existing_policy['Id']
                # Update policy
                self.organizations_client.update_policy(
                    PolicyId=policy_id,
                    Content=policy_content
                )
            else:
                # Create policy
                response = self.organizations_client.create_policy(
                    Name=policy_name,
                    Description=f"Clouptimizer {policy_type} policy",
                    Type=policy_type.upper(),
                    Content=policy_content
                )
                policy_id = response['Policy']['PolicySummary']['Id']
            
            # Attach to target
            if not target_ou:
                roots = self.organizations_client.list_roots()
                target_ou = roots['Roots'][0]['Id'] if roots['Roots'] else None
            
            if target_ou:
                self.organizations_client.attach_policy(
                    PolicyId=policy_id,
                    TargetId=target_ou
                )
                
                logger.info(f"Applied policy {policy_name} to {target_ou}")
                return True
                
        except ClientError as e:
            logger.error(f"Error applying organization policy: {e}")
        
        return False
    
    def _generate_policy_content(self, policy_name: str, policy_type: str) -> str:
        """Generate policy content based on type"""
        if policy_type == "tag_policy":
            return '''{
                "tags": {
                    "Environment": {
                        "tag_key": {
                            "@@assign": "Environment"
                        },
                        "tag_value": {
                            "@@assign": ["Production", "Staging", "Development", "Test"]
                        }
                    },
                    "Owner": {
                        "tag_key": {
                            "@@assign": "Owner"
                        }
                    },
                    "CostCenter": {
                        "tag_key": {
                            "@@assign": "CostCenter"
                        }
                    }
                }
            }'''
        else:
            # Service Control Policy example
            return '''{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Action": [
                            "ec2:TerminateInstances"
                        ],
                        "Resource": "*",
                        "Condition": {
                            "StringNotEquals": {
                                "aws:PrincipalOrgID": "${aws:PrincipalOrgID}"
                            }
                        }
                    }
                ]
            }'''
    
    def get_account_by_id(self, account_id: str) -> Optional[AWSAccount]:
        """Get account by ID"""
        return next((acc for acc in self.accounts if acc.account_id == account_id), None)
    
    def get_accounts_by_ou(self, ou_id: str) -> List[AWSAccount]:
        """Get all accounts in an organizational unit"""
        ou = next((ou for ou in self.organizational_units if ou.ou_id == ou_id), None)
        if ou:
            return [self.get_account_by_id(acc_id) for acc_id in ou.accounts 
                   if self.get_account_by_id(acc_id)]
        return []
    
    def get_account_hierarchy(self) -> Dict[str, Any]:
        """Get complete account hierarchy"""
        hierarchy = {
            'organization': {
                'master_account': next((acc.account_id for acc in self.accounts 
                                      if acc.account_type == AccountType.MANAGEMENT), None),
                'total_accounts': len(self.accounts),
                'active_accounts': sum(1 for acc in self.accounts if acc.is_active),
                'organizational_units': []
            }
        }
        
        # Build OU hierarchy
        root_ous = [ou for ou in self.organizational_units if ou.parent_id is None]
        
        for root_ou in root_ous:
            hierarchy['organization']['organizational_units'].append(
                self._build_ou_hierarchy(root_ou)
            )
        
        return hierarchy
    
    def _build_ou_hierarchy(self, ou: OrganizationalUnit) -> Dict[str, Any]:
        """Build OU hierarchy recursively"""
        ou_data = {
            'id': ou.ou_id,
            'name': ou.name,
            'account_count': ou.account_count,
            'accounts': ou.accounts,
            'policies': ou.policies,
            'children': []
        }
        
        # Add child OUs
        for child_id in ou.child_ous:
            child_ou = next((o for o in self.organizational_units if o.ou_id == child_id), None)
            if child_ou:
                ou_data['children'].append(self._build_ou_hierarchy(child_ou))
        
        return ou_data
    
    def generate_cross_account_report(self, resources_by_account: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Generate cross-account analysis report.
        
        Args:
            resources_by_account: Resources grouped by account ID
            
        Returns:
            Cross-account analysis report
        """
        report = {
            'total_accounts': len(resources_by_account),
            'total_resources': sum(len(resources) for resources in resources_by_account.values()),
            'by_account': {},
            'by_service': defaultdict(int),
            'by_region': defaultdict(int),
            'total_monthly_cost': 0,
            'top_spending_accounts': [],
            'compliance_summary': {},
            'optimization_opportunities': []
        }
        
        # Analyze each account
        for account_id, resources in resources_by_account.items():
            account = self.get_account_by_id(account_id)
            account_name = account.account_name if account else "Unknown"
            
            account_cost = sum(r.get('monthly_cost', 0) for r in resources)
            
            report['by_account'][account_id] = {
                'name': account_name,
                'resource_count': len(resources),
                'monthly_cost': account_cost,
                'services': len(set(r.get('type', 'unknown') for r in resources)),
                'regions': len(set(r.get('region', 'unknown') for r in resources))
            }
            
            report['total_monthly_cost'] += account_cost
            
            # Aggregate by service and region
            for resource in resources:
                report['by_service'][resource.get('type', 'unknown')] += 1
                report['by_region'][resource.get('region', 'unknown')] += 1
        
        # Sort top spending accounts
        report['top_spending_accounts'] = sorted(
            [{'account_id': k, **v} for k, v in report['by_account'].items()],
            key=lambda x: x['monthly_cost'],
            reverse=True
        )[:10]
        
        # Add compliance summary
        report['compliance_summary'] = {
            'accounts_with_tagging_issues': self._check_tagging_compliance(resources_by_account),
            'accounts_with_security_issues': self._check_security_compliance(resources_by_account),
            'accounts_with_cost_anomalies': self._check_cost_anomalies(resources_by_account)
        }
        
        return report
    
    def _check_tagging_compliance(self, resources_by_account: Dict[str, List[Dict]]) -> List[str]:
        """Check tagging compliance across accounts"""
        non_compliant = []
        
        for account_id, resources in resources_by_account.items():
            untagged = sum(1 for r in resources if not r.get('tags'))
            if untagged > len(resources) * 0.2:  # More than 20% untagged
                non_compliant.append(account_id)
        
        return non_compliant
    
    def _check_security_compliance(self, resources_by_account: Dict[str, List[Dict]]) -> List[str]:
        """Check security compliance across accounts"""
        # Simplified check - in production, implement comprehensive security checks
        return []
    
    def _check_cost_anomalies(self, resources_by_account: Dict[str, List[Dict]]) -> List[str]:
        """Check for cost anomalies across accounts"""
        anomalies = []
        
        if not resources_by_account:
            return anomalies
        
        # Calculate average cost per account
        costs = [sum(r.get('monthly_cost', 0) for r in resources) 
                for resources in resources_by_account.values()]
        
        if costs:
            avg_cost = sum(costs) / len(costs)
            std_cost = (sum((c - avg_cost) ** 2 for c in costs) / len(costs)) ** 0.5
            
            # Flag accounts with costs > 2 standard deviations from mean
            for account_id, resources in resources_by_account.items():
                account_cost = sum(r.get('monthly_cost', 0) for r in resources)
                if account_cost > avg_cost + 2 * std_cost:
                    anomalies.append(account_id)
        
        return anomalies