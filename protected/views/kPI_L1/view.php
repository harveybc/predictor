<?php
$this->breadcrumbs=array(
	'Kpi  L1s'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'List KPI_L1', 'url'=>array('index')),
	array('label'=>'Create KPI_L1', 'url'=>array('create')),
	array('label'=>'Update KPI_L1', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete KPI_L1', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage KPI_L1', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View KPI_L1 #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'Fecha',
		'Eff_Shift',
		'Count_Fill_Batch',
		'Count_Pall_Batch',
		'Count_Fill_Shift',
	),
)); ?>


