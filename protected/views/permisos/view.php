<?php
$this->breadcrumbs=array(
	'Permisos'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Permisos', 'url'=>array('index')),
	array('label'=>'Nuevo Permiso', 'url'=>array('create')),
	array('label'=>'Actualizar Permiso', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Permiso', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Permisos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Permisos #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'modulo0.descripcion',
		'usuario0.Username',
		'operacion0.descripcion',
		'documento0.descripcion',
	),
)); ?>


