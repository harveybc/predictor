<?php
$this->breadcrumbs=array(
	'Equipos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Equipos', 'url'=>array('index')),
	array('label'=>'Nuevo Equipo', 'url'=>array('create')),
	array('label'=>'Detalles de Equipos', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Equipos', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->Equipo))
        $nombre=$modelTMP->Equipo;
?>

<?php $this->setPageTitle (' Actualizar Equipo:<?php echo $nombre?>'); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'estructura-form',
	'enableAjaxValidation'=>true,
)); 

echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
