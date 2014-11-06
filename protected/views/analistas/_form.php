<p class="note">Los campos con <span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>
<div styler="padding-left:10px;padding-top:5px;width:353px!important;margin-left:0px;margin-bottom:5px !important;margin-top:12px !important;color:#961C1F;height:206px !important">


    <div>
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Analista'); ?>
                    <?php echo $form->textField($model, 'Analista', array('size' => 50, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'Analista'); ?>
                </div>
            </div>
        </div>

        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Proceso'); ?>
                    <?php echo $form->textField($model, 'Proceso', array('size' => 50, 'maxlength' => 50)); ?>
                    <?php echo $form->error($model, 'Proceso'); ?>
                </div>
            </div>
        </div>

        </div>
        <div>
        <div>
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Pto_trabajo'); ?>
                    <?php echo $form->textField($model, 'Pto_trabajo', array('size' => 8, 'maxlength' => 8)); ?>
                    <?php echo $form->error($model, 'Pto_trabajo'); ?>
                </div>
            </div>
       
            <div>
                <div class="row">
                    <?php echo $form->labelEx($model, 'Equipo de Trabajo'); ?>
                    <?php echo $form->textField($model, 'modulo'); ?>
                    <?php echo $form->error($model, 'modulo'); ?>
                </div>

            </div>
        </div>

    </div>

</div>









