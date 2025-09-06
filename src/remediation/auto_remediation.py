"""
Automated remediation with rollback capabilities.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class RemediationType(Enum):
    """Types of remediation actions."""
    STOP_INSTANCE = "stop_instance"
    TERMINATE_INSTANCE = "terminate_instance"
    DELETE_SNAPSHOT = "delete_snapshot"
    DELETE_VOLUME = "delete_volume"
    MODIFY_INSTANCE_TYPE = "modify_instance_type"
    DELETE_OLD_AMI = "delete_old_ami"
    RELEASE_EIP = "release_eip"
    DELETE_NAT_GATEWAY = "delete_nat_gateway"
    MODIFY_RDS_INSTANCE = "modify_rds_instance"
    DELETE_LOAD_BALANCER = "delete_load_balancer"
    RESIZE_LAMBDA = "resize_lambda"
    DELETE_LOG_GROUP = "delete_log_group"
    MODIFY_DYNAMODB_CAPACITY = "modify_dynamodb_capacity"
    DELETE_S3_OBJECTS = "delete_s3_objects"
    ENABLE_S3_LIFECYCLE = "enable_s3_lifecycle"


class RiskLevel(Enum):
    """Risk levels for remediation actions."""
    ZERO = "zero"  # No risk, safe to auto-apply
    LOW = "low"    # Minimal risk, auto-apply with notification
    MEDIUM = "medium"  # Some risk, requires approval
    HIGH = "high"  # High risk, requires manual review
    CRITICAL = "critical"  # Critical risk, requires multiple approvals


@dataclass
class RemediationAction:
    """Represents a remediation action."""
    id: str
    type: RemediationType
    resource_id: str
    resource_type: str
    description: str
    risk_level: RiskLevel
    estimated_savings: float
    parameters: Dict[str, Any]
    rollback_data: Optional[Dict[str, Any]] = None
    approval_required: bool = True
    auto_apply: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None


@dataclass
class RemediationResult:
    """Result of a remediation action."""
    action_id: str
    success: bool
    message: str
    rollback_available: bool
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoRemediationEngine:
    """Automated remediation engine with rollback capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = boto3.Session()
        self.audit_log: List[RemediationAction] = []
        self.rollback_history: Dict[str, Dict[str, Any]] = {}
        
        # Risk thresholds for auto-apply
        self.auto_apply_risk_levels = {
            RiskLevel.ZERO,
            RiskLevel.LOW if config.get('auto_apply_low_risk', False) else None
        }
        
        # Action handlers
        self.action_handlers = {
            RemediationType.STOP_INSTANCE: self._stop_instance,
            RemediationType.TERMINATE_INSTANCE: self._terminate_instance,
            RemediationType.DELETE_SNAPSHOT: self._delete_snapshot,
            RemediationType.DELETE_VOLUME: self._delete_volume,
            RemediationType.MODIFY_INSTANCE_TYPE: self._modify_instance_type,
            RemediationType.DELETE_OLD_AMI: self._delete_old_ami,
            RemediationType.RELEASE_EIP: self._release_eip,
            RemediationType.DELETE_NAT_GATEWAY: self._delete_nat_gateway,
            RemediationType.MODIFY_RDS_INSTANCE: self._modify_rds_instance,
            RemediationType.DELETE_LOAD_BALANCER: self._delete_load_balancer,
            RemediationType.RESIZE_LAMBDA: self._resize_lambda,
            RemediationType.DELETE_LOG_GROUP: self._delete_log_group,
            RemediationType.MODIFY_DYNAMODB_CAPACITY: self._modify_dynamodb_capacity,
            RemediationType.DELETE_S3_OBJECTS: self._delete_s3_objects,
            RemediationType.ENABLE_S3_LIFECYCLE: self._enable_s3_lifecycle,
        }
        
        # Rollback handlers
        self.rollback_handlers = {
            RemediationType.STOP_INSTANCE: self._rollback_stop_instance,
            RemediationType.DELETE_SNAPSHOT: self._rollback_delete_snapshot,
            RemediationType.DELETE_VOLUME: self._rollback_delete_volume,
            RemediationType.MODIFY_INSTANCE_TYPE: self._rollback_modify_instance_type,
            RemediationType.RELEASE_EIP: self._rollback_release_eip,
            RemediationType.MODIFY_RDS_INSTANCE: self._rollback_modify_rds_instance,
            RemediationType.RESIZE_LAMBDA: self._rollback_resize_lambda,
            RemediationType.MODIFY_DYNAMODB_CAPACITY: self._rollback_modify_dynamodb_capacity,
            RemediationType.ENABLE_S3_LIFECYCLE: self._rollback_s3_lifecycle,
        }
    
    def generate_remediation_plan(self, findings: List[Dict[str, Any]]) -> List[RemediationAction]:
        """Generate remediation actions from optimization findings."""
        actions = []
        
        for finding in findings:
            action = self._create_action_from_finding(finding)
            if action:
                actions.append(action)
        
        # Sort by risk level and estimated savings
        actions.sort(key=lambda x: (x.risk_level.value, -x.estimated_savings))
        
        return actions
    
    def _create_action_from_finding(self, finding: Dict[str, Any]) -> Optional[RemediationAction]:
        """Create a remediation action from an optimization finding."""
        finding_type = finding.get('type')
        
        # Map finding types to remediation actions
        action_mapping = {
            'unused_ec2': (RemediationType.TERMINATE_INSTANCE, RiskLevel.MEDIUM),
            'idle_ec2': (RemediationType.STOP_INSTANCE, RiskLevel.LOW),
            'old_snapshot': (RemediationType.DELETE_SNAPSHOT, RiskLevel.LOW),
            'unattached_volume': (RemediationType.DELETE_VOLUME, RiskLevel.LOW),
            'oversized_instance': (RemediationType.MODIFY_INSTANCE_TYPE, RiskLevel.MEDIUM),
            'old_ami': (RemediationType.DELETE_OLD_AMI, RiskLevel.LOW),
            'unused_eip': (RemediationType.RELEASE_EIP, RiskLevel.ZERO),
            'idle_nat_gateway': (RemediationType.DELETE_NAT_GATEWAY, RiskLevel.HIGH),
            'oversized_rds': (RemediationType.MODIFY_RDS_INSTANCE, RiskLevel.HIGH),
            'unused_elb': (RemediationType.DELETE_LOAD_BALANCER, RiskLevel.MEDIUM),
            'oversized_lambda': (RemediationType.RESIZE_LAMBDA, RiskLevel.LOW),
            'old_log_group': (RemediationType.DELETE_LOG_GROUP, RiskLevel.ZERO),
            'oversized_dynamodb': (RemediationType.MODIFY_DYNAMODB_CAPACITY, RiskLevel.MEDIUM),
            'old_s3_objects': (RemediationType.DELETE_S3_OBJECTS, RiskLevel.LOW),
            's3_no_lifecycle': (RemediationType.ENABLE_S3_LIFECYCLE, RiskLevel.ZERO),
        }
        
        if finding_type not in action_mapping:
            return None
        
        action_type, risk_level = action_mapping[finding_type]
        
        # Generate unique action ID
        action_id = self._generate_action_id(finding)
        
        # Determine if action can be auto-applied
        auto_apply = risk_level in self.auto_apply_risk_levels
        approval_required = risk_level not in {RiskLevel.ZERO, RiskLevel.LOW}
        
        return RemediationAction(
            id=action_id,
            type=action_type,
            resource_id=finding.get('resource_id'),
            resource_type=finding.get('resource_type'),
            description=finding.get('description', ''),
            risk_level=risk_level,
            estimated_savings=finding.get('estimated_savings', 0),
            parameters=finding.get('parameters', {}),
            approval_required=approval_required,
            auto_apply=auto_apply
        )
    
    def _generate_action_id(self, finding: Dict[str, Any]) -> str:
        """Generate unique action ID."""
        content = f"{finding.get('type')}_{finding.get('resource_id')}_{datetime.utcnow()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def execute_action(self, action: RemediationAction, dry_run: bool = False) -> RemediationResult:
        """Execute a remediation action."""
        start_time = datetime.utcnow()
        
        try:
            # Check approval status
            if action.approval_required and not self._check_approval(action):
                return RemediationResult(
                    action_id=action.id,
                    success=False,
                    message="Action requires approval",
                    rollback_available=False,
                    execution_time=0
                )
            
            # Save rollback data before execution
            if action.type in self.rollback_handlers:
                action.rollback_data = self._capture_rollback_data(action)
            
            # Execute action
            if dry_run:
                logger.info(f"DRY RUN: Would execute {action.type.value} on {action.resource_id}")
                result = RemediationResult(
                    action_id=action.id,
                    success=True,
                    message="Dry run successful",
                    rollback_available=False,
                    execution_time=0
                )
            else:
                handler = self.action_handlers.get(action.type)
                if not handler:
                    raise ValueError(f"No handler for action type: {action.type}")
                
                result = handler(action)
                action.applied_at = datetime.utcnow()
                action.status = "applied" if result.success else "failed"
            
            # Update audit log
            self.audit_log.append(action)
            
            # Store rollback data if successful
            if result.success and action.rollback_data:
                self.rollback_history[action.id] = {
                    'action': action,
                    'rollback_data': action.rollback_data,
                    'timestamp': datetime.utcnow()
                }
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action.id}: {str(e)}")
            action.status = "failed"
            action.error_message = str(e)
            
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def rollback_action(self, action_id: str) -> RemediationResult:
        """Rollback a previously executed action."""
        if action_id not in self.rollback_history:
            return RemediationResult(
                action_id=action_id,
                success=False,
                message="No rollback data available",
                rollback_available=False,
                execution_time=0
            )
        
        rollback_info = self.rollback_history[action_id]
        action = rollback_info['action']
        
        try:
            handler = self.rollback_handlers.get(action.type)
            if not handler:
                return RemediationResult(
                    action_id=action_id,
                    success=False,
                    message=f"No rollback handler for action type: {action.type}",
                    rollback_available=False,
                    execution_time=0
                )
            
            result = handler(action, rollback_info['rollback_data'])
            
            if result.success:
                action.rolled_back_at = datetime.utcnow()
                action.status = "rolled_back"
                del self.rollback_history[action_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Error rolling back action {action_id}: {str(e)}")
            return RemediationResult(
                action_id=action_id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _check_approval(self, action: RemediationAction) -> bool:
        """Check if action is approved."""
        # In production, this would check an approval system
        # For now, return True for demo purposes
        return self.config.get('auto_approve', False)
    
    def _capture_rollback_data(self, action: RemediationAction) -> Dict[str, Any]:
        """Capture data needed for rollback."""
        rollback_data = {}
        
        if action.type == RemediationType.STOP_INSTANCE:
            ec2 = self.session.client('ec2')
            response = ec2.describe_instances(InstanceIds=[action.resource_id])
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                rollback_data['state'] = instance['State']['Name']
                rollback_data['instance_type'] = instance['InstanceType']
        
        elif action.type == RemediationType.MODIFY_INSTANCE_TYPE:
            ec2 = self.session.client('ec2')
            response = ec2.describe_instances(InstanceIds=[action.resource_id])
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                rollback_data['original_type'] = instance['InstanceType']
        
        elif action.type == RemediationType.RESIZE_LAMBDA:
            lambda_client = self.session.client('lambda')
            response = lambda_client.get_function_configuration(
                FunctionName=action.resource_id
            )
            rollback_data['memory_size'] = response['MemorySize']
            rollback_data['timeout'] = response['Timeout']
        
        return rollback_data
    
    # Action handlers
    def _stop_instance(self, action: RemediationAction) -> RemediationResult:
        """Stop an EC2 instance."""
        try:
            ec2 = self.session.client('ec2')
            ec2.stop_instances(InstanceIds=[action.resource_id])
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully stopped instance {action.resource_id}",
                rollback_available=True,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _terminate_instance(self, action: RemediationAction) -> RemediationResult:
        """Terminate an EC2 instance."""
        try:
            # Create snapshot before termination for safety
            ec2 = self.session.client('ec2')
            
            # Get instance volumes
            response = ec2.describe_instances(InstanceIds=[action.resource_id])
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                volumes = [bdm['Ebs']['VolumeId'] for bdm in instance.get('BlockDeviceMappings', []) 
                          if 'Ebs' in bdm]
                
                # Create snapshots
                snapshots = []
                for volume_id in volumes:
                    snapshot = ec2.create_snapshot(
                        VolumeId=volume_id,
                        Description=f"Backup before terminating {action.resource_id}"
                    )
                    snapshots.append(snapshot['SnapshotId'])
                
                action.rollback_data = {'snapshots': snapshots}
            
            # Terminate instance
            ec2.terminate_instances(InstanceIds=[action.resource_id])
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully terminated instance {action.resource_id}",
                rollback_available=False,
                execution_time=0,
                metadata={'snapshots': snapshots if 'snapshots' in locals() else []}
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_snapshot(self, action: RemediationAction) -> RemediationResult:
        """Delete an EBS snapshot."""
        try:
            ec2 = self.session.client('ec2')
            ec2.delete_snapshot(SnapshotId=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted snapshot {action.resource_id}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_volume(self, action: RemediationAction) -> RemediationResult:
        """Delete an EBS volume."""
        try:
            ec2 = self.session.client('ec2')
            
            # Create snapshot before deletion
            snapshot = ec2.create_snapshot(
                VolumeId=action.resource_id,
                Description=f"Backup before deleting volume {action.resource_id}"
            )
            action.rollback_data = {'snapshot_id': snapshot['SnapshotId']}
            
            # Delete volume
            ec2.delete_volume(VolumeId=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted volume {action.resource_id}",
                rollback_available=True,
                execution_time=0,
                metadata={'snapshot_id': snapshot['SnapshotId']}
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _modify_instance_type(self, action: RemediationAction) -> RemediationResult:
        """Modify EC2 instance type."""
        try:
            ec2 = self.session.client('ec2')
            new_type = action.parameters.get('new_instance_type')
            
            # Stop instance if running
            response = ec2.describe_instances(InstanceIds=[action.resource_id])
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                if instance['State']['Name'] == 'running':
                    ec2.stop_instances(InstanceIds=[action.resource_id])
                    waiter = ec2.get_waiter('instance_stopped')
                    waiter.wait(InstanceIds=[action.resource_id])
            
            # Modify instance type
            ec2.modify_instance_attribute(
                InstanceId=action.resource_id,
                InstanceType={'Value': new_type}
            )
            
            # Start instance
            ec2.start_instances(InstanceIds=[action.resource_id])
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully modified instance type to {new_type}",
                rollback_available=True,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_old_ami(self, action: RemediationAction) -> RemediationResult:
        """Delete old AMI and associated snapshots."""
        try:
            ec2 = self.session.client('ec2')
            
            # Get AMI details
            response = ec2.describe_images(ImageIds=[action.resource_id])
            if response['Images']:
                ami = response['Images'][0]
                snapshot_ids = [bdm['Ebs']['SnapshotId'] for bdm in ami.get('BlockDeviceMappings', [])
                              if 'Ebs' in bdm and 'SnapshotId' in bdm['Ebs']]
            
            # Deregister AMI
            ec2.deregister_image(ImageId=action.resource_id)
            
            # Delete associated snapshots
            for snapshot_id in snapshot_ids:
                try:
                    ec2.delete_snapshot(SnapshotId=snapshot_id)
                except ClientError:
                    pass  # Continue if snapshot deletion fails
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted AMI {action.resource_id}",
                rollback_available=False,
                execution_time=0,
                metadata={'deleted_snapshots': snapshot_ids if 'snapshot_ids' in locals() else []}
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _release_eip(self, action: RemediationAction) -> RemediationResult:
        """Release an Elastic IP."""
        try:
            ec2 = self.session.client('ec2')
            ec2.release_address(AllocationId=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully released EIP {action.resource_id}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_nat_gateway(self, action: RemediationAction) -> RemediationResult:
        """Delete a NAT Gateway."""
        try:
            ec2 = self.session.client('ec2')
            ec2.delete_nat_gateway(NatGatewayId=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted NAT Gateway {action.resource_id}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _modify_rds_instance(self, action: RemediationAction) -> RemediationResult:
        """Modify RDS instance class."""
        try:
            rds = self.session.client('rds')
            new_class = action.parameters.get('new_instance_class')
            
            rds.modify_db_instance(
                DBInstanceIdentifier=action.resource_id,
                DBInstanceClass=new_class,
                ApplyImmediately=False  # Apply during maintenance window
            )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully scheduled RDS modification to {new_class}",
                rollback_available=True,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_load_balancer(self, action: RemediationAction) -> RemediationResult:
        """Delete a load balancer."""
        try:
            # Determine if it's ALB/NLB or Classic
            if action.parameters.get('type') == 'classic':
                elb = self.session.client('elb')
                elb.delete_load_balancer(LoadBalancerName=action.resource_id)
            else:
                elbv2 = self.session.client('elbv2')
                elbv2.delete_load_balancer(LoadBalancerArn=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted load balancer {action.resource_id}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _resize_lambda(self, action: RemediationAction) -> RemediationResult:
        """Resize Lambda function memory."""
        try:
            lambda_client = self.session.client('lambda')
            new_memory = action.parameters.get('new_memory_size')
            
            lambda_client.update_function_configuration(
                FunctionName=action.resource_id,
                MemorySize=new_memory
            )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully resized Lambda to {new_memory}MB",
                rollback_available=True,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_log_group(self, action: RemediationAction) -> RemediationResult:
        """Delete CloudWatch log group."""
        try:
            logs = self.session.client('logs')
            logs.delete_log_group(logGroupName=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted log group {action.resource_id}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _modify_dynamodb_capacity(self, action: RemediationAction) -> RemediationResult:
        """Modify DynamoDB table capacity."""
        try:
            dynamodb = self.session.client('dynamodb')
            
            params = {
                'TableName': action.resource_id
            }
            
            if 'read_capacity' in action.parameters:
                params['ProvisionedThroughput'] = {
                    'ReadCapacityUnits': action.parameters['read_capacity'],
                    'WriteCapacityUnits': action.parameters.get('write_capacity', 5)
                }
            
            if 'billing_mode' in action.parameters:
                params['BillingMode'] = action.parameters['billing_mode']
            
            dynamodb.update_table(**params)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully modified DynamoDB table capacity",
                rollback_available=True,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _delete_s3_objects(self, action: RemediationAction) -> RemediationResult:
        """Delete old S3 objects."""
        try:
            s3 = self.session.client('s3')
            bucket = action.parameters.get('bucket')
            prefix = action.parameters.get('prefix', '')
            days_old = action.parameters.get('days_old', 90)
            
            # List objects older than specified days
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            objects_to_delete = []
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        objects_to_delete.append({'Key': obj['Key']})
            
            # Delete objects in batches
            if objects_to_delete:
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i:i+1000]
                    s3.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': batch}
                    )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully deleted {len(objects_to_delete)} old objects",
                rollback_available=False,
                execution_time=0,
                metadata={'deleted_count': len(objects_to_delete)}
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    def _enable_s3_lifecycle(self, action: RemediationAction) -> RemediationResult:
        """Enable S3 lifecycle policy."""
        try:
            s3 = self.session.client('s3')
            bucket = action.resource_id
            
            lifecycle_policy = {
                'Rules': [
                    {
                        'ID': 'auto-remediation-lifecycle',
                        'Status': 'Enabled',
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 90,
                                'StorageClass': 'GLACIER'
                            }
                        ],
                        'Expiration': {
                            'Days': 365
                        }
                    }
                ]
            }
            
            s3.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration=lifecycle_policy
            )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully enabled lifecycle policy on {bucket}",
                rollback_available=True,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=False,
                execution_time=0
            )
    
    # Rollback handlers
    def _rollback_stop_instance(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Rollback stop instance action."""
        try:
            if rollback_data.get('state') == 'running':
                ec2 = self.session.client('ec2')
                ec2.start_instances(InstanceIds=[action.resource_id])
                
                return RemediationResult(
                    action_id=action.id,
                    success=True,
                    message=f"Successfully restarted instance {action.resource_id}",
                    rollback_available=False,
                    execution_time=0
                )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message="Instance was not running before, no rollback needed",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _rollback_delete_snapshot(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Cannot rollback snapshot deletion."""
        return RemediationResult(
            action_id=action.id,
            success=False,
            message="Cannot rollback snapshot deletion",
            rollback_available=False,
            execution_time=0
        )
    
    def _rollback_delete_volume(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Restore volume from snapshot."""
        try:
            ec2 = self.session.client('ec2')
            snapshot_id = rollback_data.get('snapshot_id')
            
            if snapshot_id:
                # Create volume from snapshot
                response = ec2.create_volume(
                    SnapshotId=snapshot_id,
                    AvailabilityZone=rollback_data.get('availability_zone', 'us-east-1a')
                )
                
                return RemediationResult(
                    action_id=action.id,
                    success=True,
                    message=f"Successfully restored volume from snapshot {snapshot_id}",
                    rollback_available=False,
                    execution_time=0,
                    metadata={'new_volume_id': response['VolumeId']}
                )
            
            return RemediationResult(
                action_id=action.id,
                success=False,
                message="No snapshot available for rollback",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _rollback_modify_instance_type(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Rollback instance type modification."""
        try:
            ec2 = self.session.client('ec2')
            original_type = rollback_data.get('original_type')
            
            # Stop instance
            ec2.stop_instances(InstanceIds=[action.resource_id])
            waiter = ec2.get_waiter('instance_stopped')
            waiter.wait(InstanceIds=[action.resource_id])
            
            # Revert instance type
            ec2.modify_instance_attribute(
                InstanceId=action.resource_id,
                InstanceType={'Value': original_type}
            )
            
            # Start instance
            ec2.start_instances(InstanceIds=[action.resource_id])
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully reverted instance type to {original_type}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _rollback_release_eip(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Cannot rollback EIP release."""
        return RemediationResult(
            action_id=action.id,
            success=False,
            message="Cannot rollback EIP release - allocate a new one if needed",
            rollback_available=False,
            execution_time=0
        )
    
    def _rollback_modify_rds_instance(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Rollback RDS instance modification."""
        try:
            rds = self.session.client('rds')
            original_class = rollback_data.get('original_class')
            
            rds.modify_db_instance(
                DBInstanceIdentifier=action.resource_id,
                DBInstanceClass=original_class,
                ApplyImmediately=True
            )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully reverted RDS instance class to {original_class}",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _rollback_resize_lambda(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Rollback Lambda function resize."""
        try:
            lambda_client = self.session.client('lambda')
            
            lambda_client.update_function_configuration(
                FunctionName=action.resource_id,
                MemorySize=rollback_data.get('memory_size'),
                Timeout=rollback_data.get('timeout')
            )
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message=f"Successfully reverted Lambda configuration",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _rollback_modify_dynamodb_capacity(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Rollback DynamoDB capacity modification."""
        try:
            dynamodb = self.session.client('dynamodb')
            
            params = {
                'TableName': action.resource_id
            }
            
            if 'original_capacity' in rollback_data:
                params['ProvisionedThroughput'] = rollback_data['original_capacity']
            
            if 'original_billing_mode' in rollback_data:
                params['BillingMode'] = rollback_data['original_billing_mode']
            
            dynamodb.update_table(**params)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message="Successfully reverted DynamoDB table configuration",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def _rollback_s3_lifecycle(self, action: RemediationAction, rollback_data: Dict[str, Any]) -> RemediationResult:
        """Rollback S3 lifecycle policy."""
        try:
            s3 = self.session.client('s3')
            
            if rollback_data.get('previous_policy'):
                # Restore previous policy
                s3.put_bucket_lifecycle_configuration(
                    Bucket=action.resource_id,
                    LifecycleConfiguration=rollback_data['previous_policy']
                )
            else:
                # Delete lifecycle policy
                s3.delete_bucket_lifecycle(Bucket=action.resource_id)
            
            return RemediationResult(
                action_id=action.id,
                success=True,
                message="Successfully reverted S3 lifecycle policy",
                rollback_available=False,
                execution_time=0
            )
        except ClientError as e:
            return RemediationResult(
                action_id=action.id,
                success=False,
                message=str(e),
                rollback_available=True,
                execution_time=0
            )
    
    def get_audit_trail(self, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit trail of remediation actions."""
        trail = []
        
        for action in self.audit_log:
            if start_date and action.created_at < start_date:
                continue
            if end_date and action.created_at > end_date:
                continue
            
            trail.append({
                'id': action.id,
                'type': action.type.value,
                'resource_id': action.resource_id,
                'risk_level': action.risk_level.value,
                'status': action.status,
                'created_at': action.created_at.isoformat(),
                'applied_at': action.applied_at.isoformat() if action.applied_at else None,
                'rolled_back_at': action.rolled_back_at.isoformat() if action.rolled_back_at else None,
                'estimated_savings': action.estimated_savings,
                'error_message': action.error_message
            })
        
        return trail
    
    def get_remediation_summary(self) -> Dict[str, Any]:
        """Get summary of remediation activities."""
        total_actions = len(self.audit_log)
        applied_actions = sum(1 for a in self.audit_log if a.status == 'applied')
        failed_actions = sum(1 for a in self.audit_log if a.status == 'failed')
        rolled_back_actions = sum(1 for a in self.audit_log if a.status == 'rolled_back')
        total_savings = sum(a.estimated_savings for a in self.audit_log if a.status == 'applied')
        
        risk_distribution = {}
        for action in self.audit_log:
            risk_level = action.risk_level.value
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        type_distribution = {}
        for action in self.audit_log:
            action_type = action.type.value
            type_distribution[action_type] = type_distribution.get(action_type, 0) + 1
        
        return {
            'total_actions': total_actions,
            'applied_actions': applied_actions,
            'failed_actions': failed_actions,
            'rolled_back_actions': rolled_back_actions,
            'success_rate': (applied_actions / total_actions * 100) if total_actions > 0 else 0,
            'total_estimated_savings': total_savings,
            'risk_distribution': risk_distribution,
            'type_distribution': type_distribution,
            'rollback_available_count': len(self.rollback_history)
        }