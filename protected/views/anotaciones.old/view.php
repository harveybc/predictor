<?php
$this->breadcrumbs=array(
	'Anotaciones'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Anotaciones', 'url'=>array('index')),
	array('label'=>'Nueva Anotación', 'url'=>array('create')),
	array('label'=>'Actualizar Anotación', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Anotación', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro de borrar esto?')),
	array('label'=>'Gestionar Anotaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle('Detalles de Anotación:<?php echo $model->descripcion; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		//'id',
		'usuario0.Username',
		'descripcion',
		'documento0.descripcion',
		'eliminado',
	),
)); ?>


