<?php
$this->breadcrumbs=array(
	'Operaciones'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Operaciones', 'url'=>array('index')),
	array('label'=>'Nueva Operación', 'url'=>array('create')),
	array('label'=>'Actualizar Operación', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Operación', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Operaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Operaciones #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle(' This Autorizaciones belongs to this Operaciones: '); ?>
<ul><?php foreach($model->autorizaciones as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->autorizado, array('autorizaciones/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' This Eventos belongs to this Operaciones: '); ?>
<ul><?php foreach($model->eventoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->id, array('eventos/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' This Permisos belongs to this Operaciones: '); ?>
<ul><?php foreach($model->permisoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->id, array('permisos/view', 'id' => $foreignobj->id)));

				} ?></ul>