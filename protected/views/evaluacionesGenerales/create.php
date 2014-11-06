<?php
$this->breadcrumbs=array(
	'Evaluaciones Generales'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Evaluaciones Generales', 'url'=>array('index')),
	array('label'=>'Gestionar Evaluaciones Generales', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Evaluaciones Generales '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'evaluaciones-generales-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
