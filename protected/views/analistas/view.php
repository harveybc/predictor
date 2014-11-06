<?php
$this->breadcrumbs=array(
	'Analistas'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Analistas', 'url'=>array('index')),
	array('label'=>'Nuevo Analista', 'url'=>array('create')),
	array('label'=>'Actualizar Analista', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Analista', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'Está seguro de borrar esto?')),
	array('label'=>'Gestionar Analistas', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Analista #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		//'id',
		'Analista',
		'Proceso',
		'Pto_trabajo',
		'modulo',
	),
)); ?>


