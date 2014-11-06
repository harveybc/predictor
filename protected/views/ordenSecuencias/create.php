<?php
$this->breadcrumbs=array(
	'Orden Secuenciases'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'Lista de OrdenSecuencias', 'url'=>array('index')),
	array('label'=>'Manage OrdenSecuencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Create OrdenSecuencias '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'orden-secuencias-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Create')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
