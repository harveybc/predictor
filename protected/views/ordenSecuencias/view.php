<?php
$this->breadcrumbs=array(
	'Orden Secuenciases'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de OrdenSecuencias', 'url'=>array('index')),
	array('label'=>'Create OrdenSecuencias', 'url'=>array('create')),
	array('label'=>'Update OrdenSecuencias', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Delete OrdenSecuencias', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Are you sure you want to delete this item?')),
	array('label'=>'Manage OrdenSecuencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('View OrdenSecuencias #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
	'attributes'=>array(
		'id',
		'secuencia0.descripcion',
		'posicion',
	),
)); ?>


<br /><?php $this->setPageTitle(' This Documentos belongs to this OrdenSecuencias: '); ?>
<ul><?php foreach($model->documentoses as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->descripcion, array('documentos/view', 'id' => $foreignobj->id)));

				} ?></ul><br /><?php $this->setPageTitle(' This MetaDocs belongs to this OrdenSecuencias: '); ?>
<ul><?php foreach($model->metaDocs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->numPedido, array('metadocs/view', 'id' => $foreignobj->id)));

				} ?></ul>