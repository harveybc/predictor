<?php
$this->breadcrumbs=array(
	'Pendientes'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'List Pendientes', 'url'=>array('index')),
	array('label'=>'Create Pendientes', 'url'=>array('create')),
	array('label'=>'Update Pendientes', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete Pendientes', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')),
	array('label'=>'Manage Pendientes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View Pendientes #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'revisado',
		'fecha_enviado',
		'fecha_revisado',
		'ruta',
		'usuario',
	),
)); ?>


