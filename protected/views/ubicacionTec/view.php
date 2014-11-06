<?php
$this->breadcrumbs=array(
	'Ubicación Técnica'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Ubicación Técnica', 'url'=>array('index')),
	array('label'=>'Nueva Ubicación Técnica', 'url'=>array('create')),
	array('label'=>'Actualizar Ubicación Técnica', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Ubicación Técnica', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Ubicación Técnica', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Ubicación Técnica #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
       'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'codigoSAP',
		'descripcion',
		'padre0.codigoSAP',
		'ubicacionTecs.codigoSAP',
		'supervisor0.Username',
	),
)); ?>


<br /><?php $this->setPageTitle(' This MetaDocs belongs to this UbicacionTec: '); ?>
<ul><?php foreach($model->metaDocs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->numPedido, array('metadocs/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' This UbicacionTec belongs to this UbicacionTec: '); ?>
<ul><?php foreach($model->ubicacionTecs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->codigoSAP, array('ubicaciontec/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' This Usuarios belongs to this UbicacionTec: '); ?>
<ul><?php foreach($model->usuarioses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->Username, array('usuarios/view', 'id' => $foreignobj->id)));

				} ?></ul>