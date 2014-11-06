<?php
$this->breadcrumbs=array(
	'Archivoses'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'List Archivos', 'url'=>array('index')),
	array('label'=>'Create Archivos', 'url'=>array('create')),
	array('label'=>'Update Archivos', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete Archivos', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage Archivos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View Archivos #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'nombre',
		'tipo',
		'tamano',
		//'contenido',
	),
)); ?>


