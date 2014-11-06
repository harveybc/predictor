<?php
$this->breadcrumbs=array(
	'Evaluaciones'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Evaluaciones', 'url'=>array('index')),
	array('label'=>'Gestionar Evaluaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Evaluaciones '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'evaluaciones-form',
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
