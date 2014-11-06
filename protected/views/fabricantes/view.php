<?php
$this->breadcrumbs=array(
	'Fabricantes'=>array('index'),
	$model->id,
);

$this->menu=array(
	array('label'=>'Lista de Fabricantes', 'url'=>array('index')),
	array('label'=>'Nuevo Fabricante', 'url'=>array('create')),
	array('label'=>'Actualizar Fabricante', 'url'=>array('update', 'id'=>$model->id)),
	array('label'=>'Borrar Fabricante', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->id),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Fabricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Fabricantes #<?php echo $model->id; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'id',
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle(' Los siguientes documentos pertenecen a este fabricante'); ?>
<ul><?php foreach($model->metaDocs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->numPedido, array('metadocs/view', 'id' => $foreignobj->id)));

				} ?></ul>