<?php
$this->breadcrumbs=array(
	'Aislamiento Fase'=>array('index'),
	$model->Toma,
);

$this->menu=array(
	array('label'=>'Lista de Mediciones', 'url'=>array('index')),
	array('label'=>'Nueva Medición', 'url'=>array('create')),
	array('label'=>'Actualizar Medición', 'url'=>array('update', 'id'=>$model->Toma)),
	array('label'=>'Borrar Medición', 'url'=>'#', 'linkOptions'=>array('submit'=>array('delete','id'=>$model->Toma),'confirm'=>'Está seguro de borrar esto?')),
	array('label'=>'Gestionar Mediciones', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=Motores::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->Motor))
        $nombre=$modelTMP->Motor;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->TAG.')';

?>

<?php $this->setPageTitle('Detalles Medición Aislamiento Fase de:<br/>'.$nombre.''); ?>


<?php $this->widget('zii.widgets.CDetailView', array(
	'data'=>$model,
       'cssFile'=>'/themes/detailview/styles.css',
	'attributes'=>array(
		'TAG',
		'Fecha',
		'A050',
		'A1',
		'B050',
		'B1',
		'C050',
		'C1',
		'OT',
	),
)); ?>


