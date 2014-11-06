<?php
$this->breadcrumbs=array(
	'Autorizaciones'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Autorizaciones', 'url'=>array('index')),
	array('label'=>'Nueva Autorización', 'url'=>array('create')),
	array('label'=>'Borrar Autorización', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage Autorizaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View Autorizaciones #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
         'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'usuario0.Username',
		'documento0.descripcion',
		'operacion0.descripcion',
		'autorizado',
	),
)); ?>


