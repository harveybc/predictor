<?php
$this->breadcrumbs=array(
	'Cervecerias'=>array('index'),
	$model->descripcion,
);

$this->menu=array(
	array('label'=>'Lista de Cervecerias', 'url'=>array('index')),
	array('label'=>'Nueva Cervecería', 'url'=>array('create')),
	array('label'=>'Actualizar Cervecería', 'url'=>array('update', 'id'=>$model->descripcion)),
	array('label'=>'Borrar Cervecería', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->descripcion),'confirm'=>'Está seguro/a de borrar esto?')),
	array('label'=>'Gestionar Cervecerias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Detalles de Cervecerias #<?php echo $model->descripcion; ?>'); ?>

<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
        'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		
		'descripcion',
	),
)); ?>


<br /><?php $this->setPageTitle(' Los siguientes documentos pertenecen a estas Cervecerias: '); ?>
<ul><?php foreach($model->metaDocs as $foreignobj) { 

				printf('<li>%s</li>', CHtml::link($foreignobj->numPedido, array('metadocs/view', 'id' => $foreignobj->id)));

				} ?></ul>