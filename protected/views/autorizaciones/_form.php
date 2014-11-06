<style type="text/css">



    .secuencias{

        width: 300px;
        
        
    }



</style>


<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>


<div styler="width:325px!important;height:312px;margin-left:0px;margin-bottom:5px !important;border-color:#961C1F;padding-top:10px;padding-right:5px;padding-left:14px; ">

    <div>
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'autorizado'); ?>
                    <?php echo $form->checkBox($model, 'autorizado'); ?>
                    <?php echo $form->error($model, 'autorizado'); ?>
                </div>
            </div>
        </div>
        <div>
            <div>
                <label for="Usuarios">La autorización se aprobó para los siguientes usuarios</label><?php
                    $this->widget('application.components.Relation', array(
                        'model' => $model,
                        'relation' => 'usuario0',
                        'fields' => 'Username',
                        'allowEmpty' => false,
                        'style' => 'dropdownlist',
                        'htmlOptions' => array(
                        'class' => 'secuencias',)
                            )
                    );
                    ?>
            </div>
        </div>
        <div>
            <div>
                <label for="Documentos">La autorización pertenece a los siguientes documentos</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $model,
                    'relation' => 'documento0',
                    'fields' => 'descripcion',
                    'allowEmpty' => false,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias',)
                        )
                );
                    ?>
            </div>
        </div>
        <div>
            <div>
                <label for="Operaciones">La autorización pertenece a las siguientes operaciones</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $model,
                    'relation' => 'operacion0',
                    'fields' => 'descripcion',
                    'allowEmpty' => true,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias',)
                        )
                );
                    ?>

            </div>
        </div>

    </div>

</div>



