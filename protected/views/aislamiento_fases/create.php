<?php
$this->breadcrumbs=array(
	'Buscar por OT'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Mediciones', 'url'=>array('index')),
	array('label'=>'Gestionar Mediciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Medición Aislamiento Fase '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'aislamiento-fases-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Guardar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
