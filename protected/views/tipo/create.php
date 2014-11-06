
<?php
    //para leer el param get y con el reconfigurar los dropdown
    $valor="";
    if  (isset($_GET['id']))
    {
        $valor=$_GET['id'];
    }
?>

<?php
$this->breadcrumbs=array(
	'Tipos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Lubricantes', 'url'=>array('index')),
	array('label'=>'Gestionar Lubricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Lubricante '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'tipo-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form,
    'valor'=>$valor,
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
