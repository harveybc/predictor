<?php
$this->breadcrumbs=array(
	'Módulos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Módulos', 'url'=>array('index')),
	array('label'=>'Gestionar Módulos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Módulos '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'modulos-form',
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
