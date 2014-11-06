<?php
$this->breadcrumbs=array(
	'Reportes'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Reportes', 'url'=>array('index')),
	array('label'=>'Nuevo Reporte', 'url'=>array('create')),
	array('label'=>'Detalles de Reporte', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Reportes', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->Equipo))
        $nombre=$modelTMP->Equipo;
if (isset($modelTMP->Fecha))
        $nombre=$nombre.' ('.$modelTMP->Fecha.')';



?>

<?php $this->setPageTitle (' Actualizar reporte de fuga de gases (Ultrasonido):'.$nombre.''); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'reportes-form',
	'enableAjaxValidation'=>true,
    'htmlOptions'=>array('enctype' => 'multipart/form-data'),
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
    'modelArchivo' => $modelArchivo,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
