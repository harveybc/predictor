<?php
$this->breadcrumbs=array(
	'Evaluaciones'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Evaluaciones', 'url'=>array('index')),
	array('label'=>'Nueva Evaluación', 'url'=>array('create')),
	array('label'=>'Actualizar Evaluaciones', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Evaluaciones', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Evaluaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Evaluaciones #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
       'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'usuario0.Username',
		'fecha',
		'evaluacionGeneral0.descripcion',
		'pregunta1',
		'pregunta2',
		'pregunta3',
		'pregunta4',
		'pregunta5',
		'pregunta6',
		'pregunta7',
		'pregunta8',
		'pregunta9',
		'pregunta10',
	),
)); ?>


