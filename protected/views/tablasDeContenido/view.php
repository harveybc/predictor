<?php
$this->breadcrumbs=array(
	'Tablas De Contenidos'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de TablasDeContenido', 'url'=>array('index')),
	array('label'=>'Create TablasDeContenido', 'url'=>array('create')),
	array('label'=>'Update TablasDeContenido', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete TablasDeContenido', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage TablasDeContenido', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View TablasDeContenido #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'indice',
		'descripcion',
		'metaDoc0.numPedido',
	),
)); ?>


