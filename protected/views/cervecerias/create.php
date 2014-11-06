<?php
$this->breadcrumbs=array(
	'Cervecerias'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Cervecerias', 'url'=>array('index')),
	array('label'=>'Gestionar Cervecerias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Cervecerias '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'cervecerias-form',
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
