<?php
$this->breadcrumbs=array(
	'Anotaciones'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'Lista de Anotaciones', 'url'=>array('index')),
	array('label'=>'Gestionar Anotaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Anotaciones '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'anotaciones-form',
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
