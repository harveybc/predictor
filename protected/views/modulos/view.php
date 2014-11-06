<?php
$this->breadcrumbs=array(
	'Módulos'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Módulos', 'url'=>array('index')),
	array('label'=>'Nuevo Módulos', 'url'=>array('create')),
	array('label'=>'Actualizar Módulos', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Módulos', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Módulos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Módulos #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle(' This Eventos belongs to this Modulos: '); ?>
<ul><?php foreach($model->eventoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->id, array('eventos/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' This Permisos belongs to this Modulos: '); ?>
<ul><?php foreach($model->permisoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->id, array('permisos/view', 'id' => $foreignobj->id)));

				} ?></ul>